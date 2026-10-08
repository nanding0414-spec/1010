#!/usr/bin/env python3
"""Feature bank building script using the configured visual feature model."""
import warnings
import argparse
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.filter_stage import run_filter_stage
from project_utils.config_logger import record_config
from src.build_feature_bank import (
    create_dirs, save_class_mapping, FeatureBank
)
from src.feature_bank_dataset import FeatureBankDataset, collate_fn
from configs.config import (
    DATASET_ROOT, OUTPUT_ROOT, DATASET_TYPE, CLS_PATH,
    FEATURE_BACKBONE,
    DATASET_CLASSES, NUM_CLASSES,
    PSEUDO_MASK_TYPE, PSEUDO_MASK_ROOT, META_DIR,
    NUM_WORKERS,
    SEED,
)
from project_utils.seed_utils import set_global_seed, seed_worker
from src.feature_signature import save_feature_signature
from src.feature_extractor_factory import build_feature_extractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-build", action="store_true",
                        help="Fail on region loading/extraction errors instead of skipping them")
    args = parser.parse_args()
    if Path(OUTPUT_ROOT).exists():
        raise FileExistsError(f'Use a new OVERRIDE_OUTPUT_ROOT; refusing to overwrite {OUTPUT_ROOT}')
    print("=" * 80)
    print(f"Feature Bank Build - {DATASET_TYPE.upper()}")
    set_global_seed(SEED)
    record_config(META_DIR, 'run_build_feature_bank.py')

    # Check for pre-generated pseudo masks
    has_pseudo_masks = False
    pseudo_root = Path(PSEUDO_MASK_ROOT)
    if pseudo_root.exists():
        if DATASET_TYPE == 'voc':
            sample_masks = list(pseudo_root.glob('*.png'))[:5]
        else:
            from configs.config import DATASET_TRAIN_SPLIT
            train_dir = pseudo_root / DATASET_TRAIN_SPLIT
            sample_masks = list(train_dir.glob('*.png'))[:5] if train_dir.exists() else []
        if sample_masks:
            has_pseudo_masks = True
            print(f"Found pre-generated pseudo masks: {PSEUDO_MASK_ROOT}")

    if has_pseudo_masks:
        print(f"Mode: pre-generated pseudo masks ({PSEUDO_MASK_TYPE})")
    else:
        print("Mode: no pseudo masks found")
    print("=" * 80)

    if not has_pseudo_masks:
        raise FileNotFoundError(
            f"Pseudo masks not found under {PSEUDO_MASK_ROOT}.\n"
            f"Run first: python tools/generate_pseudo_masks_sam3.py --dataset {DATASET_TYPE}"
        )

    create_dirs()

    print(f"\n[1/6] Using pre-generated pseudo masks ({PSEUDO_MASK_TYPE})")

    # Load the configured feature extractor. SAM3 remains the pseudo-mask model.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[2/6] Loading feature model: {FEATURE_BACKBONE}...")
    feature_extractor = build_feature_extractor(device, use_interactivity=False)

    # Save class mapping
    save_class_mapping(NUM_CLASSES, CLS_PATH)
    signature_path = save_feature_signature(META_DIR, feature_extractor.feature_signature())
    print(f"[FeatureBank] Feature signature saved to {signature_path}")

    # Scan dataset
    print(f"\n[3/6] Scanning {DATASET_TYPE.upper()} dataset (train split)...")
    dataset_root = Path(DATASET_ROOT)
    
    if DATASET_TYPE == "coco":
        from configs.config import IMAGELEVEL_JSON_PATH
        from configs.config import COCO_TRAIN_LIST
        
        imagelevel_data = None
        if Path(IMAGELEVEL_JSON_PATH).is_file():
            with open(IMAGELEVEL_JSON_PATH, 'r') as f:
                imagelevel_data = json.load(f)
            
            image_paths = []
            for img_info in imagelevel_data['images']:
                img_path = dataset_root / img_info['file_name']
                if img_path.is_file():
                    image_paths.append(img_path)
            
            print(f"Loaded {len(image_paths)} train images from {IMAGELEVEL_JSON_PATH}")
        
        elif Path(COCO_TRAIN_LIST).is_file():
            with open(COCO_TRAIN_LIST, 'r') as f:
                train_ids = [line.strip() for line in f if line.strip()]
            image_paths = []
            for img_stem in train_ids:
                img_path = dataset_root / 'images' / 'train2014' / f'{img_stem}.jpg'
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} train images from {COCO_TRAIN_LIST}")
        
        else:
            # Fallback: scan SegmentationClass/train2014
            segclass_dir = dataset_root / 'SegmentationClass' / 'train2014'
            image_paths = []
            for gt_file in sorted(segclass_dir.glob('*.png')):
                img_path = dataset_root / 'images' / 'train2014' / f'{gt_file.stem}.jpg'
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} train images from SegmentationClass/train2014")

    elif DATASET_TYPE == "ade20k":
        from configs.config import IMAGELEVEL_JSON_PATH

        if Path(IMAGELEVEL_JSON_PATH).is_file():
            with open(IMAGELEVEL_JSON_PATH, 'r') as f:
                imagelevel_data = json.load(f)

            image_paths = []
            for img_info in imagelevel_data['images']:
                img_path = dataset_root / img_info['file_name']
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} ADE20K train images from {IMAGELEVEL_JSON_PATH}")
        else:
            image_dir = dataset_root / 'images' / 'training'
            image_paths = sorted(image_dir.glob("*.jpg"))
            print(f"Loaded {len(image_paths)} ADE20K train images from {image_dir}")

    elif DATASET_TYPE == "cityscapes":
        from configs.config import IMAGELEVEL_JSON_PATH

        if Path(IMAGELEVEL_JSON_PATH).is_file():
            with open(IMAGELEVEL_JSON_PATH, 'r') as f:
                imagelevel_data = json.load(f)

            image_paths = []
            for img_info in imagelevel_data['images']:
                img_path = dataset_root / img_info['file_name']
                if img_path.is_file():
                    image_paths.append(img_path)
            print(f"Loaded {len(image_paths)} CityScapes train images from {IMAGELEVEL_JSON_PATH}")
        else:
            image_dir = dataset_root / 'leftImg8bit' / 'train'
            image_paths = sorted(image_dir.glob("*/*_leftImg8bit.png"))
            print(f"Loaded {len(image_paths)} CityScapes train images from {image_dir}")

    else:
        from configs.config import DATASET_TRAIN_SPLIT

        # Use the same explicit split as pseudo-label generation and MLP training.
        from configs.config import BUILD_IMAGE_DIR
        from src.bank_input_validation import image_index
        jpeg_dir = Path(BUILD_IMAGE_DIR)
        all_image_paths = sorted(image_index(jpeg_dir).values())
        train_list_path = dataset_root / 'ImageSets' / 'Segmentation' / f'{DATASET_TRAIN_SPLIT}.txt'
        
        train_ids = {line.strip() for line in train_list_path.read_text().splitlines() if line.strip()}
        image_paths = [p for p in all_image_paths if p.stem in train_ids]
        missing_ids = train_ids - {p.stem for p in image_paths}
        if missing_ids:
            raise FileNotFoundError(
                f"{len(missing_ids)} images from {train_list_path} are missing in {jpeg_dir}: "
                f"{sorted(missing_ids)[:5]}"
            )
        print(f"VOC split list: {train_list_path}")
        print(f"JPEGImages total: {len(all_image_paths)}; {DATASET_TRAIN_SPLIT}: {len(image_paths)}")
    
    if len(image_paths) == 0:
        print("Error: no training images found.")
        return

    feature_bank = FeatureBank()

    # Initialize DataLoader
    print("\n[4/6] Initializing data loader...")

    num_workers = NUM_WORKERS

    # Create Dataset
    dataset = FeatureBankDataset(
        image_paths=image_paths,
        strict_errors=args.strict_build,
    )

    g = torch.Generator()
    g.manual_seed(SEED)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    print(f"  Dataset size: {len(dataset)} images")

    # Extract features
    print("\n[5/6] Extracting features...")

    desc_text = f"Extract ({PSEUDO_MASK_TYPE})"
    
    # Iterate DataLoader
    for batch in tqdm(dataloader, desc=desc_text):
        if not batch:
            continue
        for img_path_str, regions in batch:
            img_name = Path(img_path_str).stem
            base_extractor = getattr(feature_extractor, "base", feature_extractor)
            failures_before = getattr(base_extractor, "_feature_failure_count", 0)
            clsid_to_feats = feature_extractor.extract_features_for_regions(
                img_path_str, regions, 0
            )
            if args.strict_build:
                if getattr(base_extractor, "_feature_failure_count", 0) != failures_before:
                    raise RuntimeError(f"Feature extraction failed for {img_path_str}")
                if sum(map(len, clsid_to_feats.values())) != len(regions):
                    raise RuntimeError(f"Region features missing for {img_path_str}")

            for cls_id, feat_list in clsid_to_feats.items():
                if cls_id == 0:
                    cls_name = 'background'
                elif 1 <= cls_id <= len(DATASET_CLASSES):
                    cls_name = DATASET_CLASSES[cls_id - 1]
                else:
                    continue
                for feat_vec in feat_list:
                    if args.strict_build and not feature_bank._is_valid_feature(feat_vec):
                        raise RuntimeError(f"Invalid feature for {img_path_str}")
                    if cls_id == 0:
                        feature_bank.add_background(feat_vec, img_name)
                    else:
                        feature_bank.add_feature(cls_name, feat_vec, img_name)

    # Save stats
    print("\n[6/6] Saving statistics...")
    feature_bank.save_stats()

    print("\nBuilding FAISS indices...")
    feature_bank.build_indices()

    print("\n[Post] Running filter stage...")
    run_filter_stage(drop_ratio=None)

    print("\n" + "=" * 80)
    print("Summary:")
    print("-" * 80)
    for cls_name, stat in sorted(feature_bank.stats.items()):
        print(f"{cls_name:20s}: {stat['count']:6d} samples")
    print("=" * 80)
    print(f"Feature bank saved to: {OUTPUT_ROOT}")
    print("Done!")


if __name__ == '__main__':
    main()
