"""SAM3 native query-based mask proposer.

This backend uses SAM3's grounding decoder directly instead of issuing a dense
grid of interactive point prompts. Each text prompt runs the model's native
200 object queries and returns the masks whose query score passes the configured
threshold. Results from multiple prompts are merged with true mask-IoU NMS.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from PIL import Image


from src.mask_proposer_sam3 import (
    _Instances,
    _Predictions,
    _nms_by_masks,
    _postprocess_small_regions,
)


def parse_query_prompts(prompts: str | Iterable[str]) -> list[str]:
    """Normalize a pipe-separated string or prompt iterable."""
    if isinstance(prompts, str):
        values = prompts.split("|")
    else:
        values = list(prompts)
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        raise ValueError("SAM3 query proposer requires at least one text prompt")
    return list(dict.fromkeys(normalized))


class SAM3QueryProposer:
    """Class-agnostic proposals from SAM3's native grounding queries."""

    def __init__(
        self,
        model,
        device=None,
        prompts: str | Iterable[str] = "object",
        confidence_threshold: float = 0.15,
        mask_nms_threshold: float = 0.7,
        min_mask_area: int = 0,
        min_mask_region_area: int = 0,
        max_mask_area_ratio: float = 0.95,
        max_masks: int = 0,
        resolution: int = 1008,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.prompts = parse_query_prompts(prompts)
        self.confidence_threshold = float(confidence_threshold)
        self.mask_nms_threshold = float(mask_nms_threshold)
        self.min_mask_area = int(min_mask_area)
        self.min_mask_region_area = int(min_mask_region_area)
        self.max_mask_area_ratio = float(max_mask_area_ratio)
        self.max_masks = int(max_masks)
        from sam3.model.sam3_image_processor import Sam3Processor
        self.processor = Sam3Processor(
            model,
            resolution=int(resolution),
            device=str(self.device),
            confidence_threshold=self.confidence_threshold,
        )

    @torch.inference_mode()
    def __call__(self, img_bgr):
        """Return query masks in the same interface as the grid proposer."""
        img_rgb = np.ascontiguousarray(img_bgr[..., ::-1])
        image = Image.fromarray(img_rgb)
        width, height = image.size
        image_area = width * height
        state = self.processor.set_image(image)

        candidate_masks = []
        candidate_scores = []
        for prompt in self.prompts:
            state = self.processor.set_text_prompt(prompt=prompt, state=state)
            masks = state.get("masks")
            scores = state.get("scores")
            if masks is None or scores is None:
                continue

            masks_np = masks.detach().cpu().numpy()
            scores_np = scores.detach().float().cpu().numpy().reshape(-1)
            if masks_np.ndim == 4 and masks_np.shape[1] == 1:
                masks_np = masks_np[:, 0]
            if masks_np.ndim == 2:
                masks_np = masks_np[None]
            if masks_np.ndim != 3 or masks_np.shape[0] != scores_np.shape[0]:
                raise RuntimeError(
                    "Unexpected SAM3 query output shapes: "
                    f"masks={masks_np.shape}, scores={scores_np.shape}"
                )

            for mask, score in zip(masks_np, scores_np):
                mask = _postprocess_small_regions(
                    np.asarray(mask, dtype=bool), self.min_mask_region_area
                )
                area = int(mask.sum())
                if area < self.min_mask_area:
                    continue
                if (
                    self.max_mask_area_ratio > 0
                    and area > image_area * self.max_mask_area_ratio
                ):
                    continue
                candidate_masks.append(mask)
                candidate_scores.append(float(score))

        if not candidate_masks:
            empty_masks = torch.zeros(
                (0, height, width), device=self.device, dtype=torch.float32
            )
            empty_scores = torch.zeros(
                (0,), device=self.device, dtype=torch.float32
            )
            return _Predictions(_Instances(empty_masks, empty_scores))

        if self.mask_nms_threshold < 1.0:
            keep = _nms_by_masks(
                candidate_masks, candidate_scores, self.mask_nms_threshold
            )
        else:
            keep = list(range(len(candidate_masks)))
        keep = sorted(keep, key=lambda index: -candidate_scores[index])
        if self.max_masks > 0:
            keep = keep[: self.max_masks]

        output_masks = np.stack([candidate_masks[index] for index in keep])
        output_scores = np.asarray(
            [candidate_scores[index] for index in keep], dtype=np.float32
        )
        return _Predictions(
            _Instances(
                torch.from_numpy(output_masks.astype(np.float32)).to(self.device),
                torch.from_numpy(output_scores).to(self.device),
            )
        )


__all__ = ["SAM3QueryProposer", "parse_query_prompts"]
