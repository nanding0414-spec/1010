#!/usr/bin/env python3
"""CPU-only feature-bank diagnostics: balanced PCA/t-SNE, centroids and class statistics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature_bank_storage import load_feature_matrix_npz


def load_class(path):
    x, ids = load_feature_matrix_npz(path)
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate sample IDs: {path}")
    valid = np.isfinite(x).all(1) & (np.linalg.norm(x, axis=1) > 1e-6)
    x = x[valid]
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    return x, ids[valid], int((~valid).sum())


def scan_bank(label, root, include_background=False):
    root = Path(root).resolve()
    signature_path = root / 'meta/feature_signature.json'
    signature = json.loads(signature_path.read_text(encoding='utf-8'))
    classes, dimensions = {}, set()
    for directory in sorted((root / 'features').iterdir()):
        if not directory.is_dir():
            continue
        if not include_background and (directory.name == 'background' or directory.name.startswith('background_')):
            continue
        paths = sorted(directory.glob('*.npz'))
        if not paths:
            continue
        if len(paths) != 1:
            raise ValueError(f"Expected one matrix per class: {directory}")
        x, ids, invalid = load_class(paths[0])
        dimensions.add(x.shape[1])
        stat = paths[0].stat()
        classes[directory.name] = dict(path=paths[0], ids=ids, count=len(ids), invalid_count=invalid,
                                      size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if dimensions != {signature['embed_dim']}:
        raise ValueError(f"Feature dimensions differ from signature: {root}")
    return dict(label=label, root=root, signature=signature, classes=classes)


def class_seed(seed, name):
    return (seed + int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')) % (2 ** 32)


def select_samples(banks, max_per_class, seed, sampling='common'):
    """Use exact shared IDs by default; class-balanced sampling avoids density bias."""
    names = sorted(set.intersection(*(set(bank['classes']) for bank in banks)))
    selected, excluded = {}, []
    for name in names:
        populations = [set(bank['classes'][name]['ids'].tolist()) for bank in banks]
        rng = np.random.default_rng(class_seed(seed, name))
        if sampling == 'common':
            pool = sorted(set.intersection(*populations))
            count = min(max_per_class, len(pool))
            ids = rng.choice(pool, count, replace=False).tolist() if count else []
            picks = [ids for _ in banks]
        else:
            count = min(max_per_class, *(len(population) for population in populations))
            picks = [rng.choice(sorted(population), count, replace=False).tolist() if count else []
                     for population in populations]
        if count < 2:
            excluded.append(name)
        else:
            selected[name] = picks
    if len(selected) < 2:
        raise ValueError('Need at least two classes with two samples each. If bank IDs differ, '
                         'try --sampling independent (not matched samples).')
    excluded += sorted(set.union(*(set(bank['classes']) for bank in banks)) - set(names))
    return selected, excluded


def load_selected(bank, selected, bank_index):
    matrices, labels, sample_ids = [], [], []
    for class_index, (name, picks) in enumerate(selected.items()):
        record = bank['classes'][name]
        stat = record['path'].stat()
        if stat.st_size != record['size'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Bank changed while collecting visualization samples')
        x, ids, _ = load_class(record['path'])
        positions = {name: i for i, name in enumerate(ids.tolist())}
        chosen = picks[bank_index]
        matrices.append(x[[positions[name] for name in chosen]])
        labels.extend([class_index] * len(chosen))
        sample_ids.extend(chosen)
    return np.vstack(matrices), np.asarray(labels), sample_ids


def cohort_statistics(x, labels, names):
    means = np.stack([x[labels == i].mean(0) for i in range(len(names))])
    lengths = np.linalg.norm(means, axis=1)
    if np.any(lengths < 1e-6):
        raise ValueError('A sampled class has a zero centroid; increase sampling or inspect its features')
    centers = means / lengths[:, None]
    similarities = centers @ centers.T
    masked = similarities.copy()
    np.fill_diagonal(masked, -np.inf)
    nearest = masked.argmax(1)
    scores = x @ centers.T
    own = scores[np.arange(len(x)), labels].copy()
    scores[np.arange(len(x)), labels] = -np.inf
    margins = own - scores.max(1)
    rows = []
    for i, name in enumerate(names):
        selected = labels == i
        rows.append(dict(class_name=name, sampled_count=int(selected.sum()),
            cohesion=float(own[selected].mean()), nearest_class=names[nearest[i]],
            nearest_centroid_cosine=float(similarities[i, nearest[i]]),
            mean_centroid_margin=float(margins[selected].mean()),
            negative_margin_fraction=float((margins[selected] < 0).mean())))
    return similarities, rows


def project(x, method, seed, perplexity=30, iterations=1000):
    from sklearn.decomposition import PCA
    if method == 'pca':
        reducer = PCA(n_components=2, svd_solver='full')
        coordinates = reducer.fit_transform(x)
        variance = np.nan_to_num(reducer.explained_variance_ratio_, nan=0).tolist()
        return coordinates, dict(explained_variance_ratio=variance)
    from sklearn.manifold import TSNE
    reduced_dim = min(50, x.shape[1], len(x) - 1)
    reduced = PCA(n_components=reduced_dim, svd_solver='full').fit_transform(x)
    actual_perplexity = min(float(perplexity), (len(x) - 1) / 3)
    options = dict(n_components=2, perplexity=actual_perplexity, init='pca',
                   learning_rate='auto', random_state=seed, metric='euclidean')
    # sklearn renamed n_iter to max_iter; support both server environments.
    iteration_key = 'max_iter' if 'max_iter' in inspect.signature(TSNE).parameters else 'n_iter'
    options[iteration_key] = iterations
    reducer = TSNE(**options)
    coordinates = reducer.fit_transform(reduced)
    return coordinates, dict(perplexity=actual_perplexity, iterations=iterations,
                             pre_pca_dimensions=reduced_dim, kl_divergence=float(reducer.kl_divergence_))


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_scatter(results, names, method, output, colors):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 6), squeeze=False)
    for axis, bank in zip(axes[0], results):
        xy, labels = bank['coordinates'][method], bank['labels']
        # Shuffle drawing order so the final class does not always cover other classes.
        order = np.random.default_rng(0).permutation(len(xy))
        axis.scatter(xy[order, 0], xy[order, 1], c=[colors[i] for i in labels[order]],
                     s=8, alpha=0.65, linewidths=0, rasterized=True)
        for i, name in enumerate(names):
            position = np.median(xy[labels == i], axis=0)
            axis.text(*position, str(i + 1), fontsize=8, ha='center', va='center',
                      bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
        extra = ''
        if method == 'pca':
            explained = sum(bank['projection_info'][method]['explained_variance_ratio'])
            extra = f' | 2-D variance {100 * explained:.1f}%'
        axis.set_title(f"{bank['label']}\n{bank['dimension']}-D; n={len(xy)}{extra}", fontsize=10)
        axis.set_xlabel(f'{method.upper()} 1')
        axis.set_ylabel(f'{method.upper()} 2')
        axis.grid(alpha=0.15)
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=colors[i],
                      markeredgecolor='none', label=f'{i+1}: {name}') for i, name in enumerate(names)]
    fig.legend(handles=handles, loc='lower center', ncol=min(5, len(names)), fontsize=8,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(f'{method.upper()} | separate projections, consistent class colors', fontsize=13)
    bottom = min(0.45, 0.10 + 0.035 * np.ceil(len(names) / 5))
    fig.tight_layout(rect=(0, bottom, 1, 0.94))
    fig.text(0.5, 0.008, 'Bank labels are pseudo-label derived. Axes/distances across panels are not aligned.',
             ha='center', fontsize=8, color='#555555')
    fig.savefig(output / f'{method}_comparison.png', dpi=180)
    plt.close(fig)


def draw_diagnostics(results, names, output):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(results), figsize=(5.4 * len(results), 6), squeeze=False,
                             layout='constrained')
    for axis, bank in zip(axes[0], results):
        matrix = bank['similarities'].copy()
        np.fill_diagonal(matrix, np.nan)
        im = axis.imshow(matrix, vmin=-1, vmax=1, cmap='coolwarm')
        axis.set_xticks(range(len(names)), names, rotation=90, fontsize=7)
        axis.set_yticks(range(len(names)), names, fontsize=7)
        axis.set_title(bank['label'], fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Cosine similarity (sampled class centroids)', shrink=0.75)
    fig.suptitle('Class-center similarity | diagonal omitted; higher means more similar')
    fig.savefig(output / 'centroid_similarity.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(names) * 0.6), 10), layout='constrained')
    width = 0.8 / len(results)
    specs = [('full_count', 'Valid bank samples (before visualization sampling)'),
             ('cohesion', 'Mean cosine to own center (sampled cohort)'),
             ('mean_centroid_margin', 'Own-center minus closest other-center cosine (sampled cohort)')]
    for axis, (field, title) in zip(axes, specs):
        for j, bank in enumerate(results):
            values = [row[field] for row in bank['statistics']]
            axis.bar(np.arange(len(names)) + (j - (len(results)-1)/2) * width, values,
                     width=width, label=bank['label'])
        axis.set_xticks(range(len(names)), names, rotation=45, ha='right', fontsize=8)
        axis.set_title(title, fontsize=11)
        axis.axhline(0, color='#666666', linewidth=0.6)
        axis.grid(axis='y', alpha=0.2)
    axes[0].legend(fontsize=9, ncol=min(4, len(results)))
    fig.savefig(output / 'class_diagnostics.png', dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bank', action='append', default=[], metavar='NAME=PATH')
    parser.add_argument('--output-dir', type=Path, required=True, help='Must be a new directory')
    parser.add_argument('--method', choices=['pca', 'tsne', 'both'], default='both')
    parser.add_argument('--max-per-class', type=int, default=100)
    parser.add_argument('--sampling', choices=['common', 'independent'], default='common')
    parser.add_argument('--include-background', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.max_per_class < 2 or args.threads < 1 or args.iterations < 250 or not 0 < args.perplexity < float('inf'):
        parser.error('Require max-per-class>=2, threads>=1, iterations>=250 and positive finite perplexity')
    bank_specs = []
    for value in args.bank:
        label, separator, path = value.partition('=')
        if not separator or not label or not path:
            parser.error('--bank must be NAME=PATH')
        bank_specs.append((label, path))
    if not bank_specs or len(bank_specs) > 4 or len({name for name, _ in bank_specs}) != len(bank_specs):
        parser.error('Provide 1-4 banks with unique names')
    output = args.output_dir.resolve()
    if output.exists():
        parser.error('Use a new output directory')
    for _, path in bank_specs:
        root = Path(path).resolve()
        if output == root or output in root.parents or root in output.parents:
            parser.error('Output directory must be separate from input banks')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import sklearn
    from threadpoolctl import threadpool_limits
    banks = [scan_bank(label, path, args.include_background) for label, path in bank_specs]
    selected, excluded = select_samples(banks, args.max_per_class, args.seed, args.sampling)
    names = list(selected)
    if excluded:
        print(f'[Sampling] Excluded missing/insufficient classes: {excluded}', flush=True)
    output.mkdir(parents=True)
    methods = ['pca', 'tsne'] if args.method == 'both' else [args.method]
    cmap = plt.get_cmap('tab20' if len(names) <= 20 else 'hsv')
    colors = [('#888888' if name.startswith('background') else cmap(i / max(1, len(names)-1)))
              for i, name in enumerate(names)]
    results, csv_rows, manifests = [], [], []
    with threadpool_limits(limits=args.threads):
        for index, bank in enumerate(banks):
            print(f"[Bank {index+1}/{len(banks)}] {bank['label']}", flush=True)
            x, labels, ids = load_selected(bank, selected, index)
            similarities, statistics = cohort_statistics(x, labels, names)
            for row in statistics:
                row['full_count'] = bank['classes'][row['class_name']]['count']
                csv_rows.append(dict(bank=bank['label'], **row))
            coordinates, info = {}, {}
            for method in methods:
                print(f'  {method}: {len(x)} samples, {x.shape[1]} dimensions', flush=True)
                coordinates[method], info[method] = project(x, method, args.seed, args.perplexity, args.iterations)
                write_csv(output / f'bank_{index+1}_{method}_coordinates.csv', [dict(
                    class_name=names[labels[i]], sample_id=ids[i], x=float(point[0]), y=float(point[1]))
                    for i, point in enumerate(coordinates[method])])
            results.append(dict(label=bank['label'], dimension=x.shape[1], labels=labels,
                coordinates=coordinates, projection_info=info, similarities=similarities, statistics=statistics))
            write_csv(output / f'bank_{index+1}_centroid_similarity.csv', [dict(
                class_name=name, **{other: float(similarities[i,j]) for j, other in enumerate(names)})
                for i, name in enumerate(names)])
            manifests.append(dict(label=bank['label'], root=str(bank['root']), signature=bank['signature'],
                projection_info=info, sampled_count=len(x),
                macro_cohesion=float(np.mean([row['cohesion'] for row in statistics])),
                macro_centroid_margin=float(np.mean([row['mean_centroid_margin'] for row in statistics])),
                files={name: {k: str(v) if isinstance(v, Path) else v for k, v in record.items() if k != 'ids'}
                       for name, record in bank['classes'].items()}))
        for method in methods:
            draw_scatter(results, names, method, output, colors)
        draw_diagnostics(results, names, output)
    write_csv(output / 'class_statistics.csv', csv_rows)
    report = dict(banks=manifests, classes=names, excluded_classes=excluded,
        sampling=args.sampling, max_per_class=args.max_per_class, seed=args.seed,
        include_background=args.include_background, threads=args.threads,
        versions=dict(numpy=np.__version__, matplotlib=matplotlib.__version__, sklearn=sklearn.__version__),
        notes=['All features are L2-normalized before analysis.',
               'Centers and margins use the sampled cohort, not all bank rows.',
               'Own samples contribute to their centroid: these are descriptive statistics, not validation accuracy.',
               'Labels come from feature-bank pseudo classes, not independent ground truth.',
               'Each bank is projected independently; plot coordinates and cluster spacing are not comparable across panels.',
               'Common sampling matches class/sample-ID pairs; independently rebuilt banks must preserve ID semantics.'])
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[DONE] {output}', flush=True)


if __name__ == '__main__':
    main()
