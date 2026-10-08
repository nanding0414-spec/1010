# -*- coding: utf-8 -*-
"""Active SAM3 pseudo labels from image-level classes; native output and strict evaluation."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 数据集布局（与 tools/generate_pseudo_masks.py 保持一致）
# ---------------------------------------------------------------------------
VOC_ROOT = Path("data/VOC2012")
COCO_ROOT = Path("data/COCO2014")
ADE20K_ROOT = Path("data/ADEChallengeData2016")
CITYSCAPES_ROOT = Path("data/CityScapes")

PSEUDO_SUBDIR = "sam3"

VOC_CLASS_CONFIG = "configs/cls_voc21.txt"
COCO_CLASS_CONFIG = "configs/cls_coco_object.txt"
ADE20K_CLASS_CONFIG = "configs/cls_ade20k.txt"
CITYSCAPES_CLASS_CONFIG = "configs/cls_city_scapes.txt"


class SAM3PseudoMaskGenerator:
    """一张图共用一次图像前向，每类可独立尝试多个文本提示。"""

    def __init__(
        self,
        dataset_type="voc",
        ckpt=None,
        conf_thr=0.5,
        use_bf16=True,
        output_dir=None,
        min_mask_area=0,
        prompt_backend="active",
        active_options=None,
        classes_file=None,
        dataset_root=None,
        device=None,
        load_model=True,
        imagelevel_json=None,
        preflight_split=None,
        image_dir=None,
        output_size_reference_dir=None,
    ):
        self.dataset_type = dataset_type.lower()
        self.prompt_backend = prompt_backend
        if prompt_backend != 'active':
            raise ValueError('Only the active prompt backend is supported')
        self.active_options = dict(active_options or {})
        self.active_proposer = None
        self.active_classes = None
        self.image_metadata = {}
        self.imagelevel_json = Path(imagelevel_json) if imagelevel_json else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.output_size_reference_dir = Path(output_size_reference_dir) if output_size_reference_dir else None
        self.output_sizes = {}
        self._prepared_samples = {}
        self.use_bf16 = use_bf16
        self.conf_thr = float(conf_thr)
        self.min_mask_area = int(min_mask_area)
        if self.min_mask_area < 0 or not 0 <= self.conf_thr <= 1:
            raise ValueError("min_mask_area must be >= 0 and conf_thr must be in [0,1]")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type not in ('cuda', 'cpu'):
            raise ValueError('device must be cpu or cuda')
        if load_model and self.device.type == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable; check your environment or use --device cpu')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            torch.cuda.set_device(self.device)
            if use_bf16 and not torch.cuda.is_bf16_supported():
                raise RuntimeError('This GPU does not support BF16; add --no-bf16')

        if self.dataset_type == "voc":
            self.dataset_root = VOC_ROOT
            cls_file = VOC_CLASS_CONFIG
        elif self.dataset_type == "coco":
            self.dataset_root = COCO_ROOT
            cls_file = COCO_CLASS_CONFIG
        elif self.dataset_type == "ade20k":
            self.dataset_root = ADE20K_ROOT
            cls_file = ADE20K_CLASS_CONFIG
        elif self.dataset_type == "cityscapes":
            self.dataset_root = CITYSCAPES_ROOT
            cls_file = CITYSCAPES_CLASS_CONFIG
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        if dataset_root is not None:
            self.dataset_root = Path(dataset_root)

        # VOC/COCO 首行是背景；每行仅取标准类别名。
        lines = [
            l for l in (Path(__file__).resolve().parents[1] / cls_file).read_text(encoding="utf-8").splitlines() if l.strip()
        ]
        if self.dataset_type in ("voc", "coco"):
            lines = lines[1:]
        self._cls_lines = lines
        self.classes = [line.split(';')[0].strip() for line in lines]
        if prompt_backend == 'active':
            from tools.proposal_demo.active_utils import align_vocabulary
            self.active_classes = align_vocabulary(self.classes, classes_file)

        subdir = 'sam3_active'
        self.output_dir = (Path(output_dir) if output_dir else
                           Path(__file__).resolve().parents[1] / 'storage/pseudo_labels' / self.dataset_type / subdir)
        self.last_prompt_report = []

        # CLI checks the label manifest and image paths before importing/loading SAM3.
        if preflight_split is not None:
            self._prepared_samples[preflight_split] = self.get_image_list_and_labels(preflight_split)

        default_ckpt = Path(__file__).resolve().parents[1] / "weights" / "sam3" / "sam3.pt"
        ckpt = ckpt or (str(default_ckpt) if default_ckpt.is_file() else None)
        self.checkpoint = str(Path(ckpt).resolve()) if ckpt else 'huggingface:facebook/sam3'
        if not load_model:
            return

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        print("=" * 80)
        print(f"SAM3 Pseudo Mask Generator")
        print(f"  Dataset:      {self.dataset_type.upper()}")
        print(f"  Prompt backend: {self.prompt_backend}")
        print(f"  Confidence:   {conf_thr}")
        print(f"  bf16:         {use_bf16}")
        print(f"  min mask area: {self.min_mask_area}")
        print(f"  Output dir:   {self.output_dir}")
        print("=" * 80)

        # 本地权重优先（weights/sam3/sam3.pt），文件不存在才回退 HF 自动下载
        default_ckpt = Path(__file__).resolve().parent.parent / "weights" / "sam3" / "sam3.pt"
        ckpt = ckpt or (str(default_ckpt) if default_ckpt.is_file() else None)
        if ckpt:
            print(f"  Checkpoint:   {ckpt}")
        else:
            print("  Checkpoint:   未找到本地权重，将从 HuggingFace 自动下载")

        model = build_sam3_image_model(
            checkpoint_path=ckpt,
            load_from_HF=(ckpt is None),
            device=self.device.type,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
        )
        model.to(self.device)
        if not self.use_bf16:
            from src.sam3_precision import use_standard_vit_mlp
            patched = use_standard_vit_mlp(model)
            print(f'  Precision: standard ViT MLP for non-BF16 inference ({patched} blocks)', flush=True)
        for p in model.parameters():
            p.requires_grad = False
        self.processor = Sam3Processor(
            model, device=str(self.device), confidence_threshold=conf_thr
        )
        if prompt_backend == 'active':
            from tools.proposal_demo.active_proposer import SAM3ActiveProposer
            with torch.inference_mode(), self._amp():
                self.active_proposer = SAM3ActiveProposer(
                    model, self.device, self.active_classes, confidence_threshold=self.conf_thr,
                    min_mask_area=self.min_mask_area, **self.active_options)

    # ------------------------------------------------------------------ #
    # 工具函数
    # ------------------------------------------------------------------ #

    def _amp(self):
        return (torch.autocast('cuda', dtype=torch.bfloat16)
                if self.use_bf16 and self.device.type == 'cuda' else nullcontext())

    def _label_to_pseudo(self, label):
        """图像级标签 -> (cls_idx, 伪标签像素值)。
        VOC/COCO: 标签 1..C 即像素值（0=背景）；ADE/Cityscapes: 标签 0..C-1，像素值 +1。
        """
        if self.dataset_type in ("voc", "coco"):
            if label == 0:
                return None
            return label - 1, label
        return label, label + 1

    def _imagelevel_json_path(self, split):
        if getattr(self, 'imagelevel_json', None) is not None:
            return self.imagelevel_json
        root = self.dataset_root
        if self.dataset_type == "voc":
            return root / "ImageSets" / "ImageLevel" / f"{split}_imagelevel.json"
        if self.dataset_type == "coco":
            split_name = "train" if "train" in split else "val"
            return root / "annotations" / f"{split_name}_imagelevel.json"
        if self.dataset_type == "ade20k":
            split_dir = self._normalize_split(split, "training", "validation")
            split_name = "train" if split_dir == "training" else "val"
            return root / "ImageSets" / "ImageLevel" / f"{split_name}_imagelevel.json"
        split_dir = self._normalize_split(split, "train", "val")
        return root / "ImageSets" / "ImageLevel" / f"{split_dir}_imagelevel.json"

    @staticmethod
    def _normalize_split(split, train_name, val_name):
        split = str(split).lower()
        if split in ("train", "training"):
            return train_name
        if split in ("val", "validation"):
            return val_name
        return split

    # ------------------------------------------------------------------ #
    # 核心：一张图 -> 伪标签图
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate_mask(self, img_path, allowed_labels):
        with Image.open(img_path) as source:
            img = source.convert("RGB")
        w, h = img.size
        seg = np.zeros((h, w), dtype=np.uint8)
        self.last_prompt_report = []
        mapped_labels = []
        for value in dict.fromkeys(int(label) for label in allowed_labels):
            mapped = self._label_to_pseudo(value)
            if mapped is None:
                continue
            if not 0 <= mapped[0] < len(self.classes):
                raise ValueError(f'Invalid image-level class ID: {value}')
            mapped_labels.append(mapped)
        if getattr(self, 'prompt_backend', 'active') == 'active':
            with self._amp():
                instances = self.active_proposer(
                    np.ascontiguousarray(np.array(img)[..., ::-1]),
                    class_indices=[i for i, _ in mapped_labels])['instances']
            masks = instances.pred_masks.cpu().numpy().astype(bool)
            class_ids = self.active_proposer.last_result['class_ids']
            # Keep the existing pseudo-label overlap rule: first allowed class wins.
            # No competing-class score comparison and no background calibration.
            for cls_idx, pseudo_value in mapped_labels:
                union = masks[class_ids == cls_idx + 1].any(axis=0)
                seg[union & (seg == 0)] = pseudo_value
            self.last_prompt_report = self.active_proposer.last_result['report']
            return seg

        raise RuntimeError('Active proposer is required')

    # ------------------------------------------------------------------ #
    # 批量生成
    # ------------------------------------------------------------------ #

    def get_image_list_and_labels(self, split):
        json_path = self._imagelevel_json_path(split)
        if not json_path.is_file():
            hint = "\nUse --imagelevel-json /path/to/existing.json if the label list is stored elsewhere."
            if self.dataset_type == 'voc' and split in ('train', 'trainaug', 'val'):
                hint += ("\nOr create it with: python tools/generate_voc_imagelevel_train_val.py "
                         f'--voc-root "{self.dataset_root}" --splits {split} --seg-mask-dir SegmentationClassAug'
                         f'\nRequires ImageSets/Segmentation/{split}.txt and the corresponding GT masks.')
            raise FileNotFoundError(f"Image-level label file not found: {json_path}{hint}")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        img_paths, label_map = [], {}
        self.image_metadata = {}
        replacement_images = None
        if getattr(self, 'image_dir', None) is not None:
            from tools.visualize_proposals import image_index
            replacement_images = image_index(self.image_dir)
        for img_info in data["images"]:
            img_path = self.dataset_root / img_info["file_name"]
            if replacement_images is not None:
                stem = Path(img_info['file_name']).stem
                if stem not in replacement_images:
                    raise FileNotFoundError(f'Image ID {stem} not found in {self.image_dir}')
                img_path = replacement_images[stem]
            if not img_path.is_file():
                raise FileNotFoundError(f"Image not found: {img_path}")
            if getattr(self, 'output_size_reference_dir', None) is not None:
                reference = self.output_size_reference_dir / Path(img_info['file_name']).name
                with Image.open(reference) as original:
                    self.output_sizes[str(img_path)] = original.size
            img_paths.append(img_path)
            label_map[str(img_info.get("img_id", ""))] = list(img_info.get("labels", []))
            label_map[str(img_path)] = list(img_info.get('labels', []))
            self.image_metadata[str(img_path)] = img_info
        print(f"Loaded {len(img_paths)} images from {json_path}")
        return img_paths, label_map

    def _output_path(self, img_path, split):
        filename = f"{img_path.stem}.png"
        if self.dataset_type == "coco":
            return self.output_dir / split / filename
        if self.dataset_type == "ade20k":
            return self.output_dir / self._normalize_split(split, "training", "validation") / filename
        if self.dataset_type == "cityscapes":
            return self.output_dir / self._normalize_split(split, "train", "val") / filename
        return self.output_dir / filename

    def generate_all(self, split="train", resume=True, *, evaluate=True, eval_only=False,
                     gt_dir=None, gt_format='auto', limit=0, image_ids=None, metadata_dir=None,
                     eval_resize_prediction_to_gt=False):
        prepared = getattr(self, '_prepared_samples', {}).pop(split, None)
        img_paths, label_map = prepared if prepared is not None else self.get_image_list_and_labels(split)
        if image_ids:
            wanted = set(image_ids)
            img_paths = [path for path in img_paths if path.stem in wanted]
            missing = wanted - {path.stem for path in img_paths}
            if missing:
                raise ValueError(f'Image IDs not found in this split: {sorted(missing)}')
        if limit:
            img_paths = img_paths[:limit]
        if not img_paths or len({str(self._output_path(p, split)) for p in img_paths}) != len(img_paths):
            raise ValueError('Selected image list is empty or has duplicate output paths')
        meta_dir = Path(metadata_dir) if metadata_dir else self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        config_path = meta_dir / f"generation_{split}.json"
        labels_json = self._imagelevel_json_path(split)
        generation_config = dict(format_version=2, dataset=self.dataset_type, split=split,
                                 conf_thr=self.conf_thr, min_mask_area=self.min_mask_area,
                                 use_bf16=self.use_bf16,
                                 dataset_root=str(self.dataset_root.resolve()),
                                 checkpoint=getattr(self, 'checkpoint', None),
                                 label_json_sha256=hashlib.sha256(labels_json.read_bytes()).hexdigest() if labels_json.is_file() else None,
                                 prompt_backend=getattr(self, 'prompt_backend', 'active'),
                                 active_options=getattr(self, 'active_options', {}),
                                 active_classes=[asdict(c) for c in (getattr(self, 'active_classes', None) or [])],
                                 prompts=list(self.classes))
        if not self.use_bf16:
            generation_config['vit_mlp_precision_policy'] = 'standard_non_bf16_v1'
        if getattr(self, 'image_dir', None) is not None:
            generation_config['image_dir'] = str(self.image_dir.resolve())
        if getattr(self, 'output_size_reference_dir', None) is not None:
            generation_config['output_size_reference_dir'] = str(self.output_size_reference_dir.resolve())
            generation_config['output_resize'] = 'nearest'
            generation_config['output_sizes'] = self.output_sizes
        # JSON normalizes tuples in vocabulary metadata, keeping resume comparisons stable.
        generation_config = json.loads(json.dumps(generation_config))
        fingerprint = hashlib.sha256(json.dumps(generation_config, sort_keys=True).encode('utf-8')).hexdigest()
        completed_path = meta_dir / f'completed_{split}.json'
        completed = json.loads(completed_path.read_text(encoding='utf-8')) if completed_path.is_file() else {}

        def save_completed():
            temporary = completed_path.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(completed, indent=2), encoding='utf-8')
            temporary.replace(completed_path)
        if not eval_only and resume and config_path.is_file():
            old_config = json.loads(config_path.read_text(encoding="utf-8"))
            if old_config != generation_config:
                raise ValueError("Pseudo generation settings changed. Use a new --output-dir or --no-resume.")
        if not eval_only and resume and not config_path.is_file() and any(self._output_path(p, split).exists() for p in img_paths):
            raise ValueError('Existing masks have no generation settings. Use a new --output-dir or --no-resume.')
        if not eval_only:
            if not resume:
                for path in img_paths:
                    completed.pop(str(path.resolve()), None)
                save_completed()
            config_path.write_text(json.dumps(generation_config, indent=2, ensure_ascii=False), encoding="utf-8")
        samples = [dict(image=path, prediction=self._output_path(path, split),
                        metadata=getattr(self, 'image_metadata', {}).get(str(path), {})) for path in img_paths]
        print(f"\n{'Evaluating' if eval_only else 'Generating'} SAM3 pseudo masks for {len(img_paths)} images (split={split})...")
        success, errors, skipped = 0, [], 0
        start = time.perf_counter()
        for sample in tqdm([] if eval_only else samples, desc="Generating pseudo masks"):
            img_path, out_path = sample['image'], sample['prediction']
            try:
                if resume and out_path.is_file() and completed.get(str(img_path.resolve())) == fingerprint:
                    success += 1
                    skipped += 1
                    continue
                labels = label_map.get(str(img_path), label_map.get(img_path.stem, []))
                seg = self.generate_mask(img_path, labels)
                target_size = getattr(self, 'output_sizes', {}).get(str(img_path))
                if target_size is not None:
                    seg = np.asarray(Image.fromarray(seg.astype(np.uint8)).resize(
                        target_size, resample=Image.Resampling.NEAREST))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = out_path.with_suffix('.tmp.png')
                Image.fromarray(seg.astype(np.uint8)).save(temporary)
                temporary.replace(out_path)
                if self.prompt_backend == 'active':
                    report_path = meta_dir / "prompt_choices" / split / f"{img_path.stem}.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps(self.last_prompt_report, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
                success += 1
                completed[str(img_path.resolve())] = fingerprint
                if success % 25 == 0:
                    save_completed()
            except Exception as e:  # noqa: BLE001
                print(f"\nFailed: {img_path.name}, error: {e}")
                sample['generation_failed'] = True
                completed.pop(str(img_path.resolve()), None)
                errors.append(dict(image=str(img_path), error=str(e)))
                if isinstance(e, RuntimeError):
                    save_completed()
                    (meta_dir / f'run_{split}.json').write_text(json.dumps(
                        dict(status='failed', selected_count=len(samples), success=success, errors=errors), indent=2),
                        encoding='utf-8')
                    raise RuntimeError(f'Generation stopped at {img_path}; see the original traceback below') from e
        result = dict(selected_count=len(samples), success=success, resumed=skipped, errors=errors,
                      generation_seconds=time.perf_counter() - start, eval_only=eval_only)
        if not eval_only:
            save_completed()
            print(f'[Generation] success={success}, resumed={skipped}, failed={len(errors)}; output={self.output_dir}')
            (meta_dir / f'run_{split}.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        if evaluate or eval_only:
            from tools.pseudo_mask_evaluation import evaluate_pseudo_masks
            result['evaluation'] = evaluate_pseudo_masks(
                samples, self.dataset_root, self.dataset_type, split, self.classes,
                meta_dir / f'miou_{split}.json', gt_dir=gt_dir, gt_format=gt_format,
                resize_prediction_to_gt=eval_resize_prediction_to_gt)
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate pseudo masks with SAM 3")
    parser.add_argument("--dataset", type=str, default="voc",
                        choices=["voc", "coco", "ade20k", "cityscapes"])
    parser.add_argument("--split", type=str, default="train",
                        help="VOC: train/trainaug/val, COCO: train2014/val2014, "
                             "ADE20K: training/validation, Cityscapes: train/val")
    parser.add_argument("--no-resume", action="store_true", help="Regenerate all masks")
    parser.add_argument("--conf-thr", type=float, default=0.5,
                        help="SAM3 confidence threshold")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Local sam3.pt checkpoint (skip HF download)")
    parser.add_argument("--no-bf16", action="store_true", help="Disable bf16 autocast")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override pseudo-label output directory")
    parser.add_argument("--min-mask-area", type=int, default=0,
                        help="Minimum instance area in original pixels; empty masks never count")
    parser.add_argument('--dataset-root', type=Path, help='Override dataset directory')
    parser.add_argument('--image-dir', type=Path, help='Override input images by filename (e.g. JPEGImages_SPAN_x2)')
    parser.add_argument('--output-size-reference-dir', type=Path, help='Resize output labels with nearest neighbor to original image sizes')
    parser.add_argument('--experiment-dir', type=Path, help='Record effective arguments, hashes and results in a new experiment directory')
    parser.add_argument('--metadata-dir', type=Path, help='Separate generation metadata directory for disjoint multi-GPU shards')
    parser.add_argument('--imagelevel-json', type=Path, help='Use an existing image-level label JSON outside dataset-root')
    parser.add_argument('--device', default=None, help='e.g. cuda:0 or cpu')
    parser.add_argument('--prompt-backend', choices=['active'], default='active', help='Active SAM3 class prompting')
    parser.add_argument('--classes-file', type=Path, help='Canonical prompts and hypernym JSON; default built-in vocabulary')
    parser.add_argument('--bucket-size', type=int, default=4)
    parser.add_argument('--preview', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--preview-resolution', type=int, default=672)
    parser.add_argument('--presence-threshold', type=float, default=0.2)
    parser.add_argument('--active-keep-ratio', type=float, default=0.4)
    parser.add_argument('--min-active-classes', type=int, default=1)
    parser.add_argument('--max-active-classes', type=int, default=0)
    parser.add_argument('--neighbor-tokens', type=int, default=2)
    parser.add_argument('--hypernym-tokens', type=int, default=2)
    parser.add_argument('--context-strength', type=float, default=1.0)
    parser.add_argument('--min-mask-region-area', type=int, default=0)
    parser.add_argument('--nms-threshold', type=float, default=0.7)
    parser.add_argument('--max-mask-area-ratio', type=float, default=0.0)
    parser.add_argument('--evaluate', action=argparse.BooleanOptionalAction, default=True,
                        help='Compute mIoU against GT after generation (default on)')
    parser.add_argument('--eval-only', action='store_true', help='Evaluate existing masks without loading SAM3')
    parser.add_argument('--gt-dir', type=Path, help='Override directory containing GT PNGs')
    parser.add_argument('--eval-resize-prediction-to-gt', action='store_true',
                        help='Evaluate input-sized PNGs by nearest resizing predictions in memory; saved PNGs stay unchanged')
    parser.add_argument('--gt-format', choices=['auto', 'internal', 'labelids', 'trainids'], default='auto')
    parser.add_argument('--limit', type=int, default=0, help='First N images; 0 processes the full split')
    parser.add_argument('--image-ids', nargs='+', help='Image stems from this split for a small experiment')
    from configs.experiment import PSEUDO
    parser.set_defaults(**PSEUDO)
    args = parser.parse_args(argv)
    if args.eval_only and not args.evaluate:
        parser.error('--eval-only cannot be combined with --no-evaluate')
    import math
    for name in ('conf_thr', 'presence_threshold', 'active_keep_ratio', 'nms_threshold', 'max_mask_area_ratio'):
        if not 0 <= getattr(args, name) <= 1:
            parser.error(f'{name} must be in [0,1]')
    for name in ('min_mask_area', 'min_mask_region_area', 'max_active_classes', 'neighbor_tokens', 'hypernym_tokens', 'limit'):
        if getattr(args, name) < 0:
            parser.error(f'{name} must be >= 0')
    if args.bucket_size < 1 or args.min_active_classes < 1:
        parser.error('bucket-size and min-active-classes must be >= 1')
    if args.max_active_classes and args.max_active_classes < args.min_active_classes:
        parser.error('max-active-classes must be 0 or >= min-active-classes')
    if not math.isfinite(args.context_strength) or args.context_strength < 0:
        parser.error('context-strength must be finite and >= 0')
    if args.preview_resolution % 28 or not 224 <= args.preview_resolution <= 1008:
        parser.error('preview-resolution must be a multiple of 28 between 224 and 1008')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.experiment_dir:
        args.experiment_dir.mkdir(parents=True, exist_ok=False)
        def sha256_file(path):
            digest = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            return digest.hexdigest()
        root = Path(__file__).resolve().parents[1]
        sources = [Path(__file__), root / 'tools/pseudo_mask_evaluation.py',
                   root / 'tools/proposal_demo/active_proposer.py', root / 'tools/proposal_demo/active_utils.py',
                   root / 'tools/proposal_demo/voc_classes.json', root / 'src/mask_proposer_sam3.py',
                   root / 'src/sam3_precision.py']
        record = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                      code_sha256={str(p.relative_to(root)): sha256_file(p) for p in sources},
                      checkpoint_sha256=sha256_file(args.ckpt) if args.ckpt else None,
                      feature_bank=None, status='started')
        (args.experiment_dir / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    active_names = ('bucket_size', 'preview', 'preview_resolution', 'presence_threshold',
                    'active_keep_ratio', 'min_active_classes', 'max_active_classes',
                    'neighbor_tokens', 'hypernym_tokens', 'context_strength',
                    'min_mask_region_area', 'max_mask_area_ratio')
    active_options = {name: getattr(args, name) for name in active_names}
    active_options['mask_nms_threshold'] = args.nms_threshold

    generator = SAM3PseudoMaskGenerator(
        dataset_type=args.dataset,
        ckpt=args.ckpt,
        conf_thr=args.conf_thr,
        use_bf16=not args.no_bf16,
        output_dir=args.output_dir,
        min_mask_area=args.min_mask_area,
        prompt_backend=args.prompt_backend, active_options=active_options if args.prompt_backend == 'active' else {},
        classes_file=args.classes_file, dataset_root=args.dataset_root, device=args.device,
        load_model=not args.eval_only,
        imagelevel_json=args.imagelevel_json, preflight_split=args.split,
        image_dir=args.image_dir, output_size_reference_dir=args.output_size_reference_dir,
    )
    result = generator.generate_all(split=args.split, resume=not args.no_resume,
                                    evaluate=args.evaluate, eval_only=args.eval_only, gt_dir=args.gt_dir,
                                    gt_format=args.gt_format, limit=args.limit, image_ids=args.image_ids,
                                    metadata_dir=args.metadata_dir,
                                    eval_resize_prediction_to_gt=args.eval_resize_prediction_to_gt)
    if args.experiment_dir:
        import shutil
        for name in (f'generation_{args.split}.json', f'miou_{args.split}.json', f'run_{args.split}.json'):
            source = (args.metadata_dir or generator.output_dir / 'meta') / name
            if source.is_file():
                shutil.copy2(source, args.experiment_dir / name)
        record['status'] = 'complete' if not result['errors'] and (not args.evaluate or result['evaluation']['status'] == 'complete') else 'failed'
        (args.experiment_dir / 'run.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return int(bool(result['errors']) or ('evaluation' in result and result['evaluation']['status'] != 'complete'))


if __name__ == "__main__":
    raise SystemExit(main())
