"""SAM3 uniform-grid POINT-prompt proposer for online instance masks.

Faithful port of the SAM automatic-mask-generation idea (gen_mask_sam3.py):

  image -> (optional multi-scale crops) -> uniform point grid per crop
        -> model.predict_inst(point prompts, multimask) -> pred-IoU & stability
  filtering -> small-region cleanup -> crop-edge filtering
        -> per-crop mask NMS -> cross-crop mask NMS -> sorted masks

This yields coherent whole-object masks and much better coverage (multi-scale
crops catch small objects), avoiding the fragmentation of tiny-box grounding
prompts.

Requires the SAM3 model to be built with enable_inst_interactivity=True.
Output contains class-agnostic instance masks and scores:
    predictions["instances"].pred_masks  -> torch.Tensor [N, H, W] float(0/1)
    predictions["instances"].scores      -> torch.Tensor [N] float (pred-IoU)
"""
import numpy as np
import torch
from PIL import Image



class _Instances:
    def __init__(self, pred_masks, scores):
        self.pred_masks = pred_masks  # [N, H, W]
        self.scores = scores          # [N]


class _Predictions:
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, key):
        if key == "instances":
            return self.instances
        raise KeyError(key)


# --------------------------------------------------------------------------- #
# helpers (faithful port of gen_mask_sam3.py)
# --------------------------------------------------------------------------- #

def _make_point_grid(width, height, points_per_side):
    xs = (np.arange(points_per_side, dtype=np.float32) + 0.5) * width / points_per_side
    ys = (np.arange(points_per_side, dtype=np.float32) + 0.5) * height / points_per_side
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
    return pts.astype(np.float32)


def _batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def _crop_points_per_side(points_per_side, layer_idx, downscale_factor):
    if layer_idx == 0:
        return points_per_side
    if downscale_factor <= 1:
        return points_per_side
    return max(1, int(points_per_side / (downscale_factor ** layer_idx)))


def _generate_crop_boxes(width, height, crop_n_layers, crop_overlap_ratio):
    boxes = [(np.array([0, 0, width, height], dtype=np.int32), 0)]
    short_side = min(width, height)
    for layer in range(1, crop_n_layers + 1):
        per_side = 2 ** layer
        overlap = int(crop_overlap_ratio * short_side * (2.0 / per_side))
        cw = int(np.ceil((width + overlap * (per_side - 1)) / per_side))
        ch = int(np.ceil((height + overlap * (per_side - 1)) / per_side))
        sw = cw - overlap
        sh = ch - overlap
        for yi in range(per_side):
            for xi in range(per_side):
                x0 = min(sw * xi, width - cw)
                y0 = min(sh * yi, height - ch)
                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(width, x0 + cw)
                y1 = min(height, y0 + ch)
                boxes.append((np.array([x0, y0, x1, y1], dtype=np.int32), layer))
    return boxes


def _uncrop_mask(mask, crop_box, full_shape):
    full = np.zeros(full_shape, dtype=bool)
    x0, y0, x1, y1 = [int(v) for v in crop_box]
    full[y0:y1, x0:x1] = mask
    return full


def _uncrop_bbox(bbox, crop_box):
    x0, y0, _, _ = crop_box.astype(np.float32)
    out = bbox.copy()
    out[[0, 2]] += x0
    out[[1, 3]] += y0
    return out


def _is_box_near_crop_edge(bbox, crop_box, full_box, atol=20.0):
    if np.array_equal(crop_box, full_box):
        return False
    x0, y0, x1, y1 = crop_box.astype(np.float32)
    fx0, fy0, fx1, fy1 = full_box.astype(np.float32)
    return bool(
        (abs(bbox[0] - x0) <= atol and x0 > fx0)
        or (abs(bbox[1] - y0) <= atol and y0 > fy0)
        or (abs(bbox[2] - x1) <= atol and x1 < fx1)
        or (abs(bbox[3] - y1) <= atol and y1 < fy1)
    )


def _stability_score(logits, mask_threshold, offset):
    high = logits > (mask_threshold + offset)
    low = logits > (mask_threshold - offset)
    low_area = int(low.sum())
    if low_area == 0:
        return 0.0
    return float(np.logical_and(high, low).sum() / low_area)


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def _box_iou_one(box_a, box_b):
    """Cheap bbox overlap test used before the more expensive mask IoU."""
    x0 = max(float(box_a[0]), float(box_b[0]))
    y0 = max(float(box_a[1]), float(box_b[1]))
    x1 = min(float(box_a[2]), float(box_b[2]))
    y1 = min(float(box_a[3]), float(box_b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    return inter / max(area_a + area_b - inter, 1e-8)


def _mask_iou(mask_a, mask_b):
    """Compute true IoU without stacking all candidate masks in memory."""
    inter = np.logical_and(mask_a, mask_b).sum(dtype=np.float64)
    union = np.logical_or(mask_a, mask_b).sum(dtype=np.float64)
    return float(inter / max(union, 1.0))


def _nms_by_masks(mask_list, scores, mask_nms_thresh):
    """Greedy NMS using actual mask IoU, preserving high-quality candidates."""
    if not mask_list:
        return []
    order = np.argsort(-np.asarray(scores))
    masks = []
    boxes = []
    valid = []
    for i in order:
        mask = np.asarray(mask_list[i], dtype=bool)
        if not mask.any():
            continue
        masks.append(mask)
        boxes.append(_mask_bbox(mask))
        valid.append(int(i))
    if not masks:
        return []
    suppressed = np.zeros(len(valid), dtype=bool)
    keep = []
    for pos in range(len(valid)):
        if suppressed[pos]:
            continue
        keep.append(valid[pos])
        if pos + 1 < len(valid):
            for other_pos in range(pos + 1, len(valid)):
                if suppressed[other_pos] or _box_iou_one(boxes[pos], boxes[other_pos]) <= 0.0:
                    continue
                if _mask_iou(masks[pos], masks[other_pos]) > mask_nms_thresh:
                    suppressed[other_pos] = True
    return keep


def _remove_small_regions(mask, min_area, mode):
    if min_area <= 0:
        return mask
    try:
        import cv2
    except ImportError:
        return mask
    target = (~mask).astype(np.uint8) if mode == "holes" else mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(target, 8)
    if n_labels <= 1:
        return mask
    out = mask.copy()
    h, w = mask.shape
    for i in range(1, n_labels):
        if int(stats[i, cv2.CC_STAT_AREA]) > min_area:
            continue
        comp = labels == i
        if mode == "holes":
            ys, xs = np.where(comp)
            touches = (np.any(xs == 0) or np.any(xs == w - 1)
                       or np.any(ys == 0) or np.any(ys == h - 1))
            if not touches:
                out[comp] = True
        else:
            out[comp] = False
    return out


def _postprocess_small_regions(mask, min_area):
    if min_area <= 0:
        return mask
    mask = _remove_small_regions(mask, min_area, "holes")
    mask = _remove_small_regions(mask, min_area, "islands")
    return mask


class SAM3GridProposer:
    """Class-agnostic mask proposer: uniform point grid (+ optional crops)."""

    def __init__(
        self,
        model,
        device=None,
        grid_size=32,
        points_per_batch=64,
        multimask_output=True,
        pred_iou_thresh=0.3,
        stability_score_thresh=0.5,
        stability_score_offset=1.0,
        mask_threshold=0.0,
        min_mask_area=0,
        min_mask_region_area=0,
        max_mask_area_ratio=0.9,
        box_nms_thresh=0.7,
        crop_n_layers=1,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        crop_nms_thresh=0.7,
        filter_crop_edge=True,
        crop_edge_tolerance=20.0,
        max_masks=0,
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = int(grid_size)
        self.points_per_batch = int(points_per_batch)
        self.multimask_output = multimask_output
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.mask_threshold = mask_threshold
        self.min_mask_area = int(min_mask_area)
        self.min_mask_region_area = int(min_mask_region_area)
        self.max_mask_area_ratio = max_mask_area_ratio
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = int(crop_n_layers)
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.crop_nms_thresh = crop_nms_thresh
        self.filter_crop_edge = filter_crop_edge
        self.crop_edge_tolerance = crop_edge_tolerance
        self.max_masks = int(max_masks)
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(
            model, device=str(self.device), confidence_threshold=0.0
        )

    @torch.inference_mode()
    def __call__(self, img_bgr):
        """img_bgr: np.uint8 HxWx3 (BGR)。"""
        img_rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        image = Image.fromarray(img_rgb)
        width, height = image.size
        full_box = np.array([0, 0, width, height], dtype=np.int32)
        full_shape = (height, width)

        crop_boxes = _generate_crop_boxes(
            width, height, self.crop_n_layers, self.crop_overlap_ratio
        )

        cand_masks = []   # bool masks (full-res)
        cand_scores = []
        cand_areas = []
        cand_boxes = []   # bbox (full-res)

        for crop_box, crop_layer in crop_boxes:
            x0, y0, x1, y1 = [int(v) for v in crop_box]
            crop_image = image.crop((x0, y0, x1, y1))
            cw, ch = crop_image.size
            inference_state = self.processor.set_image(crop_image)

            pps = _crop_points_per_side(
                self.grid_size, crop_layer, self.crop_n_points_downscale_factor
            )
            points = _make_point_grid(cw, ch, pps)

            crop_masks, crop_scores = [], []
            for batch in _batched(points, self.points_per_batch):
                point_coords = batch[:, None, :]          # (B,1,2) 像素坐标
                point_labels = np.ones((len(batch), 1), dtype=np.int64)
                logits, pred_scores, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=self.multimask_output,
                    return_logits=True,
                )
                logits = np.asarray(logits)
                pred_scores = np.asarray(pred_scores)
                if logits.ndim == 3:
                    logits = logits[None, ...]
                    pred_scores = pred_scores[None, ...]
                for b in range(logits.shape[0]):
                    for c in range(logits.shape[1]):
                        sc = float(pred_scores[b, c])
                        if sc < self.pred_iou_thresh:
                            continue
                        mlogits = logits[b, c]
                        stab = _stability_score(
                            mlogits, self.mask_threshold, self.stability_score_offset
                        )
                        if stab < self.stability_score_thresh:
                            continue
                        cm = mlogits > self.mask_threshold
                        cm = _postprocess_small_regions(cm, self.min_mask_region_area)
                        if int(cm.sum()) < self.min_mask_area:
                            continue
                        bbox_crop = _mask_bbox(cm)
                        if bbox_crop is None:
                            continue
                        bbox = _uncrop_bbox(bbox_crop, crop_box)
                        if self.filter_crop_edge and _is_box_near_crop_edge(
                            bbox, crop_box, full_box, atol=self.crop_edge_tolerance
                        ):
                            continue
                        full_mask = _uncrop_mask(cm, crop_box, full_shape)
                        area = int(full_mask.sum())
                        if self.max_mask_area_ratio > 0 and area > width * height * self.max_mask_area_ratio:
                            continue
                        crop_masks.append(full_mask)
                        crop_scores.append(sc)

            if self.box_nms_thresh < 1.0:
                keep = _nms_by_masks(crop_masks, crop_scores, self.box_nms_thresh)
            else:
                keep = list(range(len(crop_masks)))

            for i in keep:
                cand_masks.append(crop_masks[i])
                cand_scores.append(crop_scores[i])
                cand_areas.append(int(crop_masks[i].sum()))
                cand_boxes.append(_mask_bbox(crop_masks[i]))

        if not cand_masks:
            empty = torch.zeros((0,), device=self.device, dtype=torch.float32)
            return _Predictions(_Instances(empty, empty))

        # Cross-crop NMS is ranked by SAM3 quality score. Crop size is not a
        # reliable quality signal: a small crop can produce a partial mask.
        if self.crop_nms_thresh < 1.0:
            keep = _nms_by_masks(cand_masks, cand_scores, self.crop_nms_thresh)
        else:
            keep = list(range(len(cand_masks)))

        # Keep the highest-quality masks when a cap is configured.
        keep = sorted(keep, key=lambda i: -cand_scores[i])
        if self.max_masks > 0:
            keep = keep[: self.max_masks]

        out_masks = np.stack([cand_masks[i] for i in keep])
        out_scores = np.array([cand_scores[i] for i in keep])
        return _Predictions(_Instances(
            torch.from_numpy(out_masks.astype(np.float32)).to(self.device),
            torch.from_numpy(out_scores.astype(np.float32)).to(self.device),
        ))


__all__ = ["SAM3GridProposer"]
