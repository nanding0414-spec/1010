"""Frozen-SAM3 supplementary proposals for the oracle-guided R experiment.

Worker API (no GT access, no Slurm submission):
``context_worker(manifest, job)`` creates ``job/context_path`` once, storing actual
canonical main-language and neighbour/hypernym tensors and content digests.
``generate_worker`` reads that shared checkpoint and writes per-image proposals;
``query_worker`` reads ORIGINAL signed image tokens and production GMP/MLP/gallery;
``rescue_worker`` adds strict-qualified queries only to the baseline background.
``run_worker`` dispatches job['stage'] without submitting further work.

Required manifest keys: version, sam3_weights, source_signature, source_adapter,
source_mlp_bank, token_records. Token records are id -> {path, sha256,
receipt_path, receipt_sha256, geometry:{original_hw:[H,W],token_hw:[72,72]}}.
Optional image_records use {path,sha256}; image_dir otherwise supplies <id>.jpg.
Job keys: arm (R1/R2/R3), image_ids, image_dir, output_dir, context_path, device.
Query additionally uses candidates_dir; rescue uses candidates_dir, queries_dir,
baseline_dir. Outputs use <output_dir>/<id>/{proposals.npz|queries.npz|prediction.png,
complete.json}. Per-image complete markers are immutable and SHA checked on reuse.
Root orchestration is responsible for frozen source/input manifests, phase attempts,
space reservation, safe process ownership, and committing all predictions before GT.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

FORMAT = 'oracle_guided_R_v1'
RULES = dict(resolution=1008, bucket_size=4, seed=42, object_score=.15,
             probability=.5, min_area=100, max_area_ratio=.95, nms=.7,
             per_class_cap=1, rho=.1, proposal=.4, retrieval=.95, margin=.1,
             novel_fraction=.1, proposal_weight=.2, centroid_weight=.2)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(_json(value), encoding='utf-8')
    os.replace(temporary, path)


def _identity(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def _read_verified(record):
    path = Path(record['path'])
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != record['sha256']:
        raise ValueError(f'Input SHA mismatch: {path}')
    if 'bytes' in record and len(content) != record['bytes']:
        raise ValueError(f'Input size mismatch: {path}')
    return content


def _binding(manifest, job, image_id, **extra):
    return dict(format=FORMAT, version=manifest['version'], arm=job['arm'],
                image_id=image_id, code_digest=manifest.get('code_digest'),
                parameter_digest=manifest.get('parameter_digest'), rules=RULES,
                source_signature=manifest['source_signature'], **extra)


def _existing(folder, binding):
    folder = Path(folder)
    marker = folder / 'complete.json'
    if marker.is_file():
        value = json.loads(marker.read_text(encoding='utf-8'))
        if value['binding'] != binding:
            raise ValueError(f'Checkpoint signature mismatch: {folder}')
        for record in value['artifacts']:
            _read_verified(record)
        return value
    if folder.exists():
        raise ValueError(f'Incomplete attempt retained; use a new attempt: {folder}')
    return None


def _complete(folder, binding, paths, report):
    value = dict(binding=binding, artifacts=[_identity(p) for p in paths],
                 report=report, no_GT_read=True)
    _write(Path(folder) / 'complete.json', value)
    return value


def mask_hash(mask):
    from src.semantic_rescue import mask_hash as production_hash
    return production_hash(mask)


def tensor_hash(value):
    """Digest actual tensor values, including shape and dtype, not neighbour names."""
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    return hashlib.sha256(_json(dict(shape=list(value.shape), dtype=str(value.dtype))).encode()
                          + value.tobytes()).hexdigest()


def replace_main_prompt(text, replacement, class_index):
    """Replace one main-prompt slot while leaving every other slot byte-identical.

    Canonical context tensors are deliberately not accepted by this function.
    SAM3's padded text sequence length must agree; no silent token truncation.
    Supports NumPy arrays in CPU tests and real torch tensors in workers.
    """
    result = {key: value.clone() if hasattr(value, 'clone') else value.copy()
              for key, value in text.items()}
    for key in ('language_features', 'language_mask'):
        axis = 0 if key == 'language_mask' else 1
        original = text[key]
        incoming = replacement[key]
        expected = list(original.shape)
        expected[axis] = 1
        if list(incoming.shape) != expected:
            raise ValueError(f'Alias text shape differs for {key}: {incoming.shape} vs {expected}')
        if axis == 0:
            result[key][class_index:class_index + 1] = incoming
        else:
            result[key][:, class_index:class_index + 1] = incoming
    return result


def center_crop_box(height, width):
    if min(height, width) < 1:
        raise ValueError('Empty image')
    h, w = max(1, math.floor(.8 * height)), max(1, math.floor(.8 * width))
    y, x = (height - h) // 2, (width - w) // 2
    return x, y, x + w, y + h


def map_crop_masks(masks, box, original_shape):
    x0, y0, x1, y1 = map(int, box)
    h, w = map(int, original_shape)
    masks = np.asarray(masks)
    if (masks.ndim != 3 or masks.shape[1:] != (y1 - y0, x1 - x0)
            or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)
            or not np.isin(masks, [0, 1]).all()):
        raise ValueError('Crop mapping geometry differs')
    output = np.zeros((len(masks), h, w), bool)
    output[:, y0:y1, x0:x1] = masks.astype(bool)
    truncated = ((masks[:, 0, :].any(1) | masks[:, -1, :].any(1)
                  | masks[:, :, 0].any(1) | masks[:, :, -1].any(1)))
    return output, truncated


def filter_and_cap(masks, scores, source_classes, candidate_ids, *, min_area=100,
                   max_area_ratio=.95, nms_iou=.7, per_class_cap=1):
    """Original-coordinate area/NMS, then global exact-hash dedup and class cap.

    Ties use immutable candidate ID, independent of shard or input ordering.
    Pre-cap geometry is returned for the independent GT-only coverage diagnostic.
    """
    masks = np.asarray(masks)
    scores = np.asarray(scores, np.float32)
    classes = np.asarray(source_classes, np.int16)
    ids = np.asarray(candidate_ids, dtype=str)
    if (masks.ndim != 3 or not np.isin(masks, [0, 1]).all()
            or scores.shape != (len(masks),) or classes.shape != scores.shape
            or ids.shape != scores.shape or not np.isfinite(scores).all()
            or (scores < 0).any() or (scores > 1).any()
            or not np.isin(classes, range(1, 21)).all()
            or len(set(ids.tolist())) != len(ids)):
        raise ValueError('Invalid candidate alignment, score, source class or mask')
    masks = masks.astype(bool, copy=False)
    areas = masks.sum((1, 2))
    ranked = sorted(range(len(masks)), key=lambda i: (-float(scores[i]), str(ids[i])))
    valid = [i for i in ranked if areas[i] >= min_area and areas[i] <= masks.shape[1] * masks.shape[2] * max_area_ratio]
    nms_kept, by_class = [], {}
    for i in valid:
        prior = by_class.setdefault(int(classes[i]), [])
        if any(np.count_nonzero(masks[i] & masks[j]) / max(np.count_nonzero(masks[i] | masks[j]), 1) > nms_iou for j in prior):
            continue
        prior.append(i)
        nms_kept.append(i)
    unique, seen = [], set()
    for i in nms_kept:
        digest = mask_hash(masks[i])
        if digest not in seen:
            seen.add(digest)
            unique.append(i)
    capped, counts = [], {}
    for i in unique:
        cls = int(classes[i])
        if counts.get(cls, 0) < per_class_cap:
            capped.append(i)
            counts[cls] = counts.get(cls, 0) + 1
    arrays = dict(masks=masks[capped], scores=scores[capped], source_classes=classes[capped],
                  candidate_ids=ids[capped], mask_hashes=np.asarray([mask_hash(masks[i]) for i in capped]),
                  masks_pre_cap=masks[unique], scores_pre_cap=scores[unique],
                  source_classes_pre_cap=classes[unique], candidate_ids_pre_cap=ids[unique],
                  mask_hashes_pre_cap=np.asarray([mask_hash(masks[i]) for i in unique]))
    stats = dict(decoded_threshold_candidates=len(masks), after_area=len(valid), after_class_nms=len(nms_kept),
                 after_hash_dedup=len(unique), before_class_cap=len(unique), after_class_cap=len(capped),
                 pre_cap_per_source_class={str(c): int((classes[unique] == c).sum()) for c in range(1, 21)},
                 post_cap_per_source_class={str(c): int((classes[capped] == c).sum()) for c in range(1, 21)})
    return arrays, stats


def _classes(manifest):
    from tools.proposal_demo.active_utils import load_vocabulary
    default = Path(__file__).resolve().parent / 'proposal_demo' / 'voc_classes.json'
    canonical = load_vocabulary(default)
    path = Path(manifest.get('vocabulary_path', default))
    values = load_vocabulary(path)
    if values != canonical or len(values) != 20 or values[10].prompt != 'dining table':
        raise ValueError('R requires the complete unmodified canonical VOC vocabulary')
    return values


def _build_model(manifest, job):
    import torch
    from sam3.model_builder import build_sam3_image_model
    from src.sam3_precision import use_standard_vit_mlp
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device(job.get('device', 'cuda:0'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_sam3_image_model(checkpoint_path=str(Path(manifest['sam3_weights']).resolve()),
                                  load_from_HF=False, device=device.type, eval_mode=True,
                                  enable_segmentation=True, enable_inst_interactivity=False).to(device).eval()
    if not use_standard_vit_mlp(model):
        raise ValueError('FP32 standard ViT MLP patch failed before first image forward')
    model.float().requires_grad_(False)
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('SAM3 is not fully FP32')
    return model, device


def context_worker(manifest, job):
    """Freeze canonical tensors once. No image forward or GT access occurs here."""
    import torch
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    path = Path(job['context_path'])
    metadata = path.with_suffix(path.suffix + '.json')
    classes = _classes(manifest)
    binding = dict(format=FORMAT, version=manifest['version'],
                   sam3_sha256=manifest.get('sam3_sha256'),
                   vocabulary=[dict(name=c.name, prompt=c.prompt, hypernyms=list(c.hypernyms)) for c in classes],
                   code_digest=manifest.get('code_digest'))
    if path.exists() or metadata.exists():
        if not path.is_file() or not metadata.is_file():
            raise ValueError('Incomplete canonical context checkpoint retained')
        receipt = json.loads(metadata.read_text(encoding='utf-8'))
        if receipt['binding'] != binding:
            raise ValueError('Canonical context binding differs')
        _read_verified(receipt['artifact'])
        return receipt
    model, device = _build_model(manifest, job)
    proposer = SAM3ActiveProposer(model, device, classes, preview=False, bucket_size=4,
                                 neighbor_tokens=2, hypernym_tokens=2, context_strength=1.,
                                 force_presence_one=False)
    payload = dict(text={k: v.detach().cpu() for k, v in proposer.text.items()},
                   context=proposer.context.detach().cpu(),
                   context_mask=proposer.context_mask.detach().cpu(), expansion=proposer.expansion)
    digests = {k: tensor_hash(v) for k, v in payload['text'].items()}
    digests.update(context=tensor_hash(payload['context']), context_mask=tensor_hash(payload['context_mask']))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or metadata.exists():
        raise ValueError('Concurrent canonical context writer; use exactly one context job')
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('xb') as stream:
        torch.save(payload, stream)
    os.replace(temporary, path)
    receipt = dict(binding=binding, artifact=_identity(path), tensor_sha256=digests,
                   encoding='actual_FP32_canonical_neighbour_hypernym_tensors', no_GT_read=True)
    _write(metadata, receipt)
    return receipt


def _frozen_proposer(manifest, job):
    import torch
    from tools.proposal_demo.active_proposer import SAM3ActiveProposer
    path = Path(job['context_path'])
    receipt = json.loads(path.with_suffix(path.suffix + '.json').read_text(encoding='utf-8'))
    content = _read_verified(receipt['artifact'])
    frozen = torch.load(io.BytesIO(content), map_location='cpu', weights_only=True)
    actual = {k: tensor_hash(v) for k, v in frozen['text'].items()}
    actual.update(context=tensor_hash(frozen['context']), context_mask=tensor_hash(frozen['context_mask']))
    if actual != receipt['tensor_sha256']:
        raise ValueError('Canonical context tensor content differs')
    if receipt['binding']['version'] != manifest['version'] or receipt['binding']['code_digest'] != manifest.get('code_digest'):
        raise ValueError('Canonical context version/code differs')
    classes = _classes(manifest)
    model, device = _build_model(manifest, job)

    class FrozenProposer(SAM3ActiveProposer):
        def _prepare_text(self, *unused):
            self.text = {k: v.clone().to(device) for k, v in frozen['text'].items()}
            self.context = frozen['context'].clone().to(device)
            self.context_mask = frozen['context_mask'].clone().to(device)
            self.expansion = frozen['expansion']

    proposer = FrozenProposer(model, device, classes, preview=False, bucket_size=4,
                              force_presence_one=False, confidence_threshold=.15,
                              mask_probability_threshold=.5)
    if job['arm'] == 'R2':
        alias = proposer._encode_text(['table'])
        proposer.text = replace_main_prompt(proposer.text, alias, 10)
    if tensor_hash(proposer.context) != actual['context'] or tensor_hash(proposer.context_mask) != actual['context_mask']:
        raise ValueError('R2 changed frozen canonical context')
    return proposer, receipt


def decode_image(proposer, rgb, image_id, arm):
    """Real SAM3 grounding: one 1008 encoder, exactly five full buckets, no preview."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    if arm not in ('R1', 'R2', 'R3'):
        raise ValueError('Unknown R arm')
    h, w = rgb.shape[:2]
    box = center_crop_box(h, w) if arm == 'R3' else (0, 0, w, h)
    x0, y0, x1, y1 = box
    view = Image.fromarray(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    masks, scores, classes, identifiers = [], [], [], []
    raw_count = 0
    with torch.inference_mode(), torch.autocast(device_type=proposer.device.type, enabled=False):
        state = proposer.processor.set_image(view)
        for indices in proposer._buckets(list(range(20))):
            out = proposer._ground(state, indices, expand=True)
            presence = out['presence_logit_dec'].sigmoid().reshape(len(indices))
            confidence = out['pred_logits'].sigmoid().squeeze(-1) * presence[:, None]
            for slot, source_class in enumerate(indices):
                selected = torch.where(confidence[slot] > .15)[0]
                raw_count += len(selected)
                for chunk in selected.split(16):
                    if not chunk.numel():
                        continue
                    probability = F.interpolate(out['pred_masks'][slot, chunk, None].float(),
                                                size=(y1-y0, x1-x0), mode='bilinear', align_corners=False).sigmoid()[:, 0]
                    masks.extend((probability > .5).cpu().numpy())
                    scores.extend(confidence[slot, chunk].float().cpu().tolist())
                    classes.extend([source_class + 1] * len(chunk))
                    identifiers.extend(f'{arm}:{image_id}:{source_class+1:02d}:{int(q):04d}' for q in chunk.cpu().tolist())
            del out
        del state
    masks = np.stack(masks) if masks else np.zeros((0, y1-y0, x1-x0), bool)
    masks, truncated = map_crop_masks(masks, box, (h, w))
    arrays, stats = filter_and_cap(masks, scores, classes, identifiers)
    truncation = dict(zip(identifiers, map(bool, truncated)))
    stats.update(encoder_calls=1, decoder_bucket_calls=5, preview_calls=0,
                 decoded_class_ids=list(range(1, 21)), raw_threshold_candidates=raw_count,
                 crop_box_xyxy=list(box), original_hw=[h, w], presence='predicted_normal',
                 force_presence_one=False, precision='fp32', resolution=1008,
                 crop_boundary_contact_before_cap=sum(truncation[i] for i in arrays['candidate_ids_pre_cap']),
                 crop_boundary_contact_after_cap=sum(truncation[i] for i in arrays['candidate_ids']))
    arrays['crop_boundary_contact'] = np.asarray([truncation[i] for i in arrays['candidate_ids']], bool)
    arrays['crop_boundary_contact_pre_cap'] = np.asarray([truncation[i] for i in arrays['candidate_ids_pre_cap']], bool)
    return arrays, stats


def _image_record(manifest, job, image_id):
    if image_id in manifest.get('image_records', {}):
        return manifest['image_records'][image_id]
    path = Path(job.get('image_dir', manifest.get('image_dir', ''))) / (image_id + '.jpg')
    return _identity(path)


def generate_worker(manifest, job):
    from PIL import Image
    start = time.monotonic()
    proposer, context = _frozen_proposer(manifest, job)
    import torch
    if proposer.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(proposer.device)
    reports = {}
    for number, image_id in enumerate(job['image_ids'], 1):
        image_record = _image_record(manifest, job, image_id)
        binding = _binding(manifest, job, image_id, image=image_record,
                           context_sha256=context['artifact']['sha256'], context_tensor_sha256=context['tensor_sha256'])
        folder = Path(job['output_dir']) / image_id
        previous = _existing(folder, binding)
        if previous:
            reports[image_id] = previous['report']
            continue
        image = np.asarray(Image.open(io.BytesIO(_read_verified(image_record))).convert('RGB'))
        image_start = time.monotonic()
        arrays, report = decode_image(proposer, image, image_id, job['arm'])
        original_hashes = set(manifest.get('original_mask_hashes', {}).get(image_id, []))
        report['hash_dedup_scope'] = 'new_generated_pool_only'
        report['duplicates_original_pre_cap'] = sum(str(k) in original_hashes for k in arrays['mask_hashes_pre_cap'])
        report['duplicates_original_post_cap'] = sum(str(k) in original_hashes for k in arrays['mask_hashes'])
        report['original_hash_diagnostic_available'] = image_id in manifest.get('original_mask_hashes', {})
        report['seconds'] = time.monotonic() - image_start
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / 'proposals.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        _complete(folder, binding, [path], report)
        reports[image_id] = report
        print(f'[R generation] {job["arm"]} {number}/{len(job["image_ids"])} {image_id} pre={report["before_class_cap"]} post={report["after_class_cap"]}', flush=True)
    peak = torch.cuda.max_memory_allocated(proposer.device) / 2**20 if proposer.device.type == 'cuda' else 0.
    result = dict(images=len(reports), arm=job['arm'], seconds=time.monotonic()-start,
                  peak_allocated_MiB=peak, reports=reports, no_GT_read=True,
                  context_sha256=context['artifact']['sha256'], context_tensor_sha256=context['tensor_sha256'])
    _write(Path(job['output_dir']) / 'worker_report.json', result)
    del proposer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def load_original_tokens(manifest, image_id, expected_shape):
    """Content-SHA once on consumption; no directory-wide or duplicate token pass."""
    record = manifest['token_records'][image_id]
    receipt_bytes = _read_verified(dict(path=record['receipt_path'], sha256=record['receipt_sha256']))
    receipt = json.loads(receipt_bytes)
    signature = {k: v for k, v in manifest['source_signature'].items() if k not in ('aggregation', 'adapter')}
    if (receipt['binding']['encoder_signature'] != signature
            or receipt['binding']['original_shape'] != list(expected_shape)
            or receipt['sha256'] != record['sha256']
            or record['geometry']['original_hw'] != list(expected_shape)
            or record['geometry']['token_hw'] != [72, 72]):
        raise ValueError('Original token receipt/source signature/geometry differs')
    content = _read_verified(record)
    with np.load(io.BytesIO(content), allow_pickle=False) as values:
        tokens = values['tokens'].copy()
        grid = values['grid'].tolist()
    if tokens.dtype != np.float32 or tokens.shape != (72, 72, 1280) or grid != [72, 72] or not np.isfinite(tokens).all():
        raise ValueError('Invalid original FP32 SAM3 token cache')
    return tokens.reshape(5184, 1280)


def _retriever(manifest, device):
    import torch
    from src.feature_adapter import load_adapter, project_features
    from src.inference import load_class_mapping, load_class_features
    from src.retrieval import build_retrieval_context, classify_queries
    mapping, foreground, background = load_class_mapping(str(Path(manifest['source_mlp_bank']) / 'meta'))
    members = load_class_features(str(Path(manifest['source_mlp_bank']) / 'features'), list(dict.fromkeys(foreground + background)))
    gallery = np.vstack(list(members.values()))
    labels = [label for label, values in members.items() for _ in values]
    gallery = np.ascontiguousarray(gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-8), np.float32)
    context = build_retrieval_context(gallery, labels)
    adapter, _ = load_adapter(manifest['source_adapter'], device, manifest['source_signature'])
    inverse = {name: key for key, name in mapping.items()}
    try:
        import faiss
        index = faiss.IndexFlatIP(gallery.shape[1])
        index.add(gallery)
    except ImportError:
        index = None
    if len(mapping) != 21 or set(mapping) != set(range(21)):
        raise ValueError('Expected original 21 semantic classes')

    def classify(raw_features):
        z = project_features(adapter, raw_features, device)
        z = np.ascontiguousarray(z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8), np.float32)
        result = classify_queries(z, gallery, labels, index, context, 25, 'balanced', 5, 2, .2)
        return z, [dict(class_id=inverse[p.label], retrieval_score=p.score, margin=p.margin,
                        class_scores=[dict(zip(context.labels, p.class_scores))[mapping[c]] for c in range(21)]) for p in result]
    return classify


def query_worker(manifest, job):
    import torch
    from src.hard_center_gmp import aggregate
    start = time.monotonic()
    device = torch.device(job.get('device', 'cuda:0'))
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    classify = _retriever(manifest, device)
    count = 0
    for number, image_id in enumerate(job['image_ids'], 1):
        proposal_folder = Path(job['candidates_dir']) / image_id
        marker = json.loads((proposal_folder / 'complete.json').read_text(encoding='utf-8'))
        source = marker['artifacts'][0]
        binding = _binding(manifest, job, image_id, proposals=source, tokens=manifest['token_records'][image_id],
                           source_adapter=manifest['source_adapter'], source_mlp_bank=manifest['source_mlp_bank'])
        folder = Path(job['output_dir']) / image_id
        previous = _existing(folder, binding)
        if previous:
            count += previous['report']['queries']
            continue
        with np.load(io.BytesIO(_read_verified(source)), allow_pickle=False) as values:
            masks = values['masks'].astype(bool)
            ids, hashes = values['candidate_ids'], values['mask_hashes']
            scores, source_classes = values['scores'], values['source_classes']
        tokens = load_original_tokens(manifest, image_id, masks.shape[1:]) if len(masks) else None
        with torch.inference_mode():
            if tokens is not None:
                t = torch.from_numpy(tokens).to(device)
                vectors = [aggregate(t, mask, rho=.1).cpu().numpy() for mask in masks]
            else:
                vectors = []
        raw = np.stack(vectors) if vectors else np.zeros((0, 1280), np.float32)
        projected, decisions = classify(raw) if len(raw) else (np.zeros((0, 256), np.float32), [])
        labels = np.asarray([d['class_id'] for d in decisions], np.int16)
        retrieval = np.asarray([d['retrieval_score'] for d in decisions], np.float32)
        margins = np.asarray([d['margin'] for d in decisions], np.float32)
        class_scores = np.asarray([d['class_scores'] for d in decisions], np.float32).reshape(-1, 21)
        fused = .8 * retrieval + .2 * scores
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / 'queries.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, raw_features=raw, projected_features=projected, class_ids=labels,
                                retrieval_scores=retrieval, margins=margins, class_scores=class_scores,
                                fused_scores=fused, proposal_scores=scores, source_classes=source_classes,
                                candidate_ids=ids, mask_hashes=hashes)
        report = dict(queries=len(raw), GMP_from_original_signed_tokens=True, production_GMP=True,
                      production_balanced_classifier=True, source_class_used_as_semantic_label=False,
                      full_class_ids=labels.tolist(), original_token_content_SHA_verified=bool(len(masks)))
        _complete(folder, binding, [path], report)
        count += len(raw)
        print(f'[R query] {job["arm"]} {number}/{len(job["image_ids"])} {image_id} queries={len(raw)}', flush=True)
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.
    result = dict(images=len(job['image_ids']), queries=count, seconds=time.monotonic()-start,
                  peak_allocated_MiB=peak, no_GT_read=True)
    _write(Path(job['output_dir']) / 'worker_report.json', result)
    return result


def strict_background_rescue(baseline, masks, query):
    """Only predicted retrieval classes matter; source_class is provenance only."""
    baseline = np.asarray(baseline)
    masks = np.asarray(masks)
    if baseline.ndim != 2 or not np.isin(baseline, range(21)).all() or masks.shape != (len(query['class_ids']), *baseline.shape):
        raise ValueError('Invalid baseline/proposal shape')
    if not np.isin(masks, [0, 1]).all():
        raise ValueError('R rescue masks must remain binary')
    for field in ('proposal_scores', 'retrieval_scores', 'margins', 'fused_scores'):
        values = np.asarray(query[field])
        if values.shape != (len(masks),) or not np.isfinite(values).all():
            raise ValueError('Invalid R rescue score alignment or finite values')
    result = baseline.astype(np.uint8, copy=True)
    occupied = result != 0
    order = sorted(range(len(masks)), key=lambda i: (-float(query['fused_scores'][i]), str(query['candidate_ids'][i])))
    accepted, added = [], 0
    for i in order:
        cls = int(query['class_ids'][i])
        if (cls <= 0 or cls > 20 or float(query['proposal_scores'][i]) < .4
                or float(query['retrieval_scores'][i]) < .95 or float(query['margins'][i]) < .1):
            continue
        mask = masks[i].astype(bool)
        delta = mask & ~occupied
        if not delta.any() or delta.sum() / max(int(mask.sum()), 1) < .1:
            continue
        result[delta] = cls
        occupied |= delta
        pixels = int(delta.sum())
        added += pixels
        accepted.append(dict(candidate_id=str(query['candidate_ids'][i]), class_id=cls,
                             source_class=int(query['source_classes'][i]), added_pixels=pixels,
                             original_mask_pixels=int(mask.sum()), novel_fraction=float(pixels / int(mask.sum()))))
    if not np.array_equal(result[baseline != 0], baseline[baseline != 0]):
        raise AssertionError('R overwrote original foreground')
    return result, dict(accepted=len(accepted), added_pixels=added, accepted_candidates=accepted)


def rescue_worker(manifest, job):
    from PIL import Image
    start = time.monotonic()
    reports = {}
    for image_id in job['image_ids']:
        baseline = Path(job['baseline_dir']) / (image_id + '.png')
        proposal_marker = json.loads((Path(job['candidates_dir']) / image_id / 'complete.json').read_text(encoding='utf-8'))
        query_marker = json.loads((Path(job['queries_dir']) / image_id / 'complete.json').read_text(encoding='utf-8'))
        proposal_record, query_record = proposal_marker['artifacts'][0], query_marker['artifacts'][0]
        base_record = manifest.get('baseline_records', {}).get(image_id)
        if base_record is None:
            base_record = _identity(baseline)
        binding = _binding(manifest, job, image_id, baseline=base_record, proposals=proposal_record, queries=query_record)
        folder = Path(job['output_dir']) / image_id
        previous = _existing(folder, binding)
        if previous:
            reports[image_id] = previous['report']
            continue
        with np.load(io.BytesIO(_read_verified(proposal_record)), allow_pickle=False) as data:
            masks, ids, hashes = data['masks'], data['candidate_ids'], data['mask_hashes']
        with np.load(io.BytesIO(_read_verified(query_record)), allow_pickle=False) as data:
            query = {key: data[key] for key in data.files}
        if not np.array_equal(query['candidate_ids'], ids) or not np.array_equal(query['mask_hashes'], hashes):
            raise ValueError('R mask/query identity mismatch')
        prediction = np.asarray(Image.open(io.BytesIO(_read_verified(base_record))))
        output, report = strict_background_rescue(prediction, masks, query)
        folder.mkdir(parents=True, exist_ok=False)
        path = folder / 'prediction.png'
        Image.fromarray(output).save(path)
        _complete(folder, binding, [path], report)
        reports[image_id] = report
    result = dict(images=len(reports), seconds=time.monotonic()-start, reports=reports, no_GT_read=True,
                  old_foreground_preserved=True)
    _write(Path(job['output_dir']) / 'worker_report.json', result)
    return result


def run_worker(manifest, job):
    """Root-owned orchestration dispatch; supports stage=context/generate/query/rescue."""
    functions = dict(context=context_worker, generate=generate_worker,
                     query=query_worker, rescue=rescue_worker)
    stage = job.get('stage', 'generate')
    if stage in functions:
        return functions[stage](manifest, job)
    raise ValueError(f'Unsupported R worker stage: {stage}')
