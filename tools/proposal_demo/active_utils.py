"""Vocabulary and preview selection shared by proposal and pseudo-label scripts.

ActiveSAM reference: https://github.com/VILA-Lab/ActiveSAM
See docs/activesam_reference.md for attribution and implementation differences.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PromptClass:
    name: str
    prompt: str
    hypernyms: tuple[str, ...] = ()


def normalize_prompt(text):
    return ' '.join(text.lower().replace('_', ' ').replace('-', ' ').split())


def load_vocabulary(path):
    """Load explicit class names, canonical prompts and optional context words."""
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(data, list) or not data:
        raise ValueError('classes-file must contain a non-empty JSON list')
    classes = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str):
            raise ValueError('Every vocabulary entry needs a string name')
        name = row['name'].strip()
        prompt = row.get('prompt', name)
        hypernyms = row.get('hypernyms', [])
        if not name or not isinstance(prompt, str) or not normalize_prompt(prompt):
            raise ValueError('Class name and prompt must not be empty')
        if name.lower() == 'background':
            raise ValueError('Do not include background; class ID 0 is reserved')
        if not isinstance(hypernyms, list) or any(not isinstance(h, str) for h in hypernyms):
            raise ValueError('hypernyms must be a list of strings')
        classes.append(PromptClass(name, normalize_prompt(prompt), tuple(dict.fromkeys(
            normalize_prompt(h) for h in hypernyms if normalize_prompt(h)))))
    if len({c.name.lower() for c in classes}) != len(classes):
        raise ValueError('Class names must be unique')
    if len({c.prompt for c in classes}) != len(classes):
        raise ValueError('Canonical prompts must be unique; merge aliases into one class')
    if len(classes) > 65534:
        raise ValueError('At most 65534 foreground classes are supported')
    return classes


def select_active_classes(scores, threshold, keep_ratio, min_classes, max_classes):
    """Keep the union of threshold hits and top fraction, then enforce the cap.

    This recall-oriented rule deliberately differs from ActiveSAM's quantile-
    only rule. All-zero presence retains min_classes and reports them as fallback.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Expected finite presence scores [C]')
    order = np.argsort(-scores, kind='stable')
    minimum = min(min_classes, len(scores))
    if scores.max() <= 0:
        return np.sort(order[:minimum]).tolist()
    target = max(minimum, int(np.ceil(len(scores) * keep_ratio)))
    selected = set(order[:target].tolist()) | set(np.flatnonzero(scores >= threshold).tolist())
    ranked = [int(i) for i in order if int(i) in selected]
    if max_classes:
        ranked = ranked[:max_classes]
    return sorted(ranked)


def align_vocabulary(class_names, path=None):
    """Keep dataset IDs/names, while borrowing canonical prompts and context.

    Custom vocabularies must cover every dataset class. Built-in VOC entries
    can also enrich matching names in another dataset; other names normalize only.
    """
    aliases = {'ship': 'boat', 'table': 'diningtable', 'people': 'person'}
    source = load_vocabulary(path or Path(__file__).with_name('voc_classes.json'))
    lookup = {normalize_prompt(c.name): c for c in source}
    result = []
    for name in class_names:
        key = normalize_prompt(name)
        entry = lookup.get(key) or lookup.get(aliases.get(key, key))
        if entry is None and path:
            raise ValueError(f'classes-file has no entry for dataset class {name!r}')
        result.append(PromptClass(name, entry.prompt if entry else normalize_prompt(name),
                                  entry.hypernyms if entry else ()))
    return result
