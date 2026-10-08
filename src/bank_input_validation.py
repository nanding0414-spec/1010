"""Validate native-resolution pseudo masks against their actual build images."""
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_index(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
            if path.stem in result:
                raise ValueError(f'Duplicate image ID: {result[path.stem]} and {path}')
            result[path.stem] = path
    return result


def native_image_dir(pseudo):
    configs = []
    for path in sorted((Path(pseudo) / 'meta').rglob('*.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(value, dict) and value.get('image_dir'):
            if value.get('output_size_reference_dir'):
                raise ValueError(f'Pseudo masks are configured for resized output: {path}')
            configs.append(Path(value['image_dir']).resolve())
    if not configs or len(set(configs)) != 1:
        raise ValueError('Cannot uniquely recover native images from pseudo metadata; pass --build-image-dir')
    return configs[0]


def validate_pairs(pseudo, images, ids):
    index = image_index(images)
    pairs = []
    for stem in sorted(ids):
        mask = Path(pseudo) / f'{stem}.png'
        if stem not in index or not mask.is_file():
            raise FileNotFoundError(f'Missing training image or pseudo mask: {stem}; {images}; {pseudo}')
        with Image.open(index[stem]) as image, Image.open(mask) as label:
            if image.size != label.size:
                raise ValueError(f'Image/mask dimensions differ for {stem}: {image.size} vs {label.size}')
            array = np.asarray(label)
            if array.ndim != 2 or not np.isin(array, list(range(21)) + [255]).all():
                raise ValueError(f'Invalid VOC class-ID PNG: {mask}')
        pairs.append(index[stem])
    if not pairs:
        raise ValueError('Empty training split')
    return pairs
