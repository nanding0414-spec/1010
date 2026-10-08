"""ActiveSAM-inspired text proposals shared by the demo and pseudo-label tool.

Reference: VILA-Lab/ActiveSAM, commit 41cad43aa340d15876d2dcf1eeda92355fc21695.
See docs/activesam_reference.md and tools/proposal_demo/ActiveSAM_LICENSE.txt.
"""

from contextlib import contextmanager
import math
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from tools.proposal_demo.active_utils import select_active_classes


@contextmanager
def preview_rope(model, resolution):
    """Temporarily resize only global ViT RoPE buffers on this model instance.

    Window attention retains its original 24x24 positions. All original buffers
    are restored even on failure; other models and full-resolution runs are unaffected.
    """
    saved = []
    try:
        for block in model.backbone.modules():
            if getattr(block, 'window_size', None) != 0 or not hasattr(block, 'attn'):
                continue
            attn = block.attn
            if not getattr(attn, 'use_rope', False):
                continue
            if getattr(attn, 'use_ve_rope', False) or getattr(attn, 'rope_tiled', False):
                raise RuntimeError('Preview requires the bundled SAM3 image ViT RoPE implementation')
            side = resolution // 14  # Bundled SAM3 image model uses patch size 14.
            count = side * side + int(getattr(attn, 'cls_token', False))
            if attn.freqs_cis.shape[0] == count:
                continue
            names = ['freqs_cis']
            if getattr(attn, 'use_rope_real', False):
                names += ['freqs_cis_real', 'freqs_cis_imag']
            saved.append((attn, {name: getattr(attn, name) for name in names}))
            scale = attn.rope_pt_size[0] / side if attn.rope_interp else 1.0
            freqs = attn.compute_cis(end_x=side, end_y=side, scale_pos=scale)
            if getattr(attn, 'cls_token', False):
                freqs = torch.cat((torch.ones_like(freqs[:1]), freqs), dim=0)
            attn.freqs_cis = freqs.to(attn.freqs_cis.device)
            if getattr(attn, 'use_rope_real', False):
                attn.freqs_cis_real = attn.freqs_cis.real
                attn.freqs_cis_imag = attn.freqs_cis.imag
        yield
    finally:
        for attn, buffers in reversed(saved):
            for name, value in buffers.items():
                setattr(attn, name, value)


@contextmanager
def skip_segmentation_head(model):
    """Run grounding presence prediction without allocating preview masks."""
    original = vars(model).get('_run_segmentation_heads')
    had_override = '_run_segmentation_heads' in vars(model)
    model._run_segmentation_heads = lambda *args, **kwargs: None
    try:
        yield
    finally:
        if had_override:
            model._run_segmentation_heads = original
        else:
            del model._run_segmentation_heads


class SAM3ActiveProposer:
    def __init__(self, model, device, classes, *, bucket_size=4,
                 preview=True, preview_resolution=672, presence_threshold=0.2,
                 active_keep_ratio=0.4, min_active_classes=3, max_active_classes=12,
                 neighbor_tokens=2, hypernym_tokens=2, context_strength=1.0,
                 confidence_threshold=0.15,
                 mask_nms_threshold=0.7, min_mask_area=100,
                 min_mask_region_area=0, max_mask_area_ratio=0.95, max_masks=0):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model, self.device, self.classes = model, device, classes
        self.bucket_size = bucket_size
        self.preview = preview
        self.preview_resolution = preview_resolution
        self.presence_threshold = presence_threshold
        self.active_keep_ratio = active_keep_ratio
        self.min_active_classes, self.max_active_classes = min_active_classes, max_active_classes
        self.confidence_threshold = confidence_threshold
        self.mask_nms_threshold = mask_nms_threshold
        self.min_mask_area, self.min_mask_region_area = min_mask_area, min_mask_region_area
        self.max_mask_area_ratio, self.max_masks = max_mask_area_ratio, max_masks
        self.processor = Sam3Processor(model, resolution=1008, device=str(device))
        self.preview_processor = Sam3Processor(model, resolution=preview_resolution, device=str(device))
        self.last_result = None
        self._prepare_text(neighbor_tokens, hypernym_tokens, context_strength)

    def _buckets(self, indices):
        for start in range(0, len(indices), self.bucket_size):
            yield indices[start:start + self.bucket_size]

    @torch.inference_mode()
    def _encode_text(self, words):
        # Bound text-encoder memory for custom large vocabularies too.
        outputs = [self.model.backbone.forward_text(words[i:i + self.bucket_size], device=self.device)
                   for i in range(0, len(words), self.bucket_size)]
        return {key: torch.cat([out[key] for out in outputs], dim=0 if key == 'language_mask' else 1)
                for key in ('language_features', 'language_mask', 'language_embeds')}

    @staticmethod
    def _pool(text):
        valid = (~text['language_mask']).T.unsqueeze(-1)
        return (text['language_embeds'].float() * valid).sum(0) / valid.sum(0).clamp_min(1)

    @torch.inference_mode()
    def _prepare_text(self, neighbor_tokens, hypernym_tokens, strength):
        text = self._encode_text([c.prompt for c in self.classes])
        pooled = self._pool(text)
        resizer = self.model.backbone.language_backbone.resizer
        self.text = {k: v for k, v in text.items() if k != 'language_embeds'}
        tokens, masks = [], []
        self.expansion = [dict(name=c.name, prompt=c.prompt, neighbors=[], hypernyms=[])
                          for c in self.classes]
        count = min(neighbor_tokens, len(self.classes) - 1)
        if count and strength:
            unit = F.normalize(pooled, dim=-1)
            similarity = unit @ unit.T
            similarity.fill_diagonal_(-float('inf'))
            values, indices = similarity.topk(count, dim=1)
            weighted = pooled[indices] * values.softmax(dim=1).unsqueeze(-1)
            extra = resizer(weighted.to(resizer.weight.dtype)).float() * strength
            tokens.append(extra.transpose(0, 1))
            masks.append(torch.zeros(len(self.classes), count, dtype=torch.bool, device=self.device))
            for i, neighbors in enumerate(indices.tolist()):
                self.expansion[i]['neighbors'] = [self.classes[j].name for j in neighbors]
        hypernyms = [list(c.hypernyms[:hypernym_tokens]) for c in self.classes]
        unique = list(dict.fromkeys(word for words in hypernyms for word in words))
        if unique and strength:
            context = self._pool(self._encode_text(unique))
            context = resizer(context.to(resizer.weight.dtype)).float() * strength
            mapping = {word: i for i, word in enumerate(unique)}
            count = max(map(len, hypernyms))
            extra = torch.zeros(count, len(self.classes), context.shape[-1], device=self.device)
            padding = torch.ones(len(self.classes), count, dtype=torch.bool, device=self.device)
            for i, words in enumerate(hypernyms):
                self.expansion[i]['hypernyms'] = words
                for j, word in enumerate(words):
                    extra[j, i] = context[mapping[word]]
                    padding[i, j] = False
            tokens.append(extra)
            masks.append(padding)
        self.context = torch.cat(tokens, dim=0) if tokens else None
        self.context_mask = torch.cat(masks, dim=1) if masks else None

    def _ground(self, state, indices, expand):
        from sam3.model.data_misc import FindStage

        count = len(indices)
        # A fresh shallow dictionary avoids accumulating context across buckets.
        backbone = dict(state['backbone_out'])
        lang = self.text['language_features'][:, indices]
        padding = self.text['language_mask'][indices]
        if expand and self.context is not None:
            lang = torch.cat((lang, self.context[:, indices].to(lang.dtype)), dim=0)
            padding = torch.cat((padding, self.context_mask[indices]), dim=1)
        backbone.update(language_features=lang, language_mask=padding)
        find = FindStage(
            img_ids=torch.zeros(count, device=self.device, dtype=torch.long),
            text_ids=torch.arange(count, device=self.device),
            input_boxes=None, input_boxes_mask=None, input_boxes_label=None,
            input_points=None, input_points_mask=None)
        return self.model.forward_grounding(
            backbone_out=backbone, find_input=find, find_target=None,
            geometric_prompt=self.model._get_dummy_prompt(num_prompts=count))

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _preview(self, image, candidates):
        self._sync()
        start = time.perf_counter()
        if not self.preview or len(candidates) <= self.min_active_classes:
            return candidates, None, 0.0, 0
        presence = np.full(len(self.classes), np.nan, dtype=np.float32)
        with preview_rope(self.model, self.preview_resolution), skip_segmentation_head(self.model):
            state = self.preview_processor.set_image(image)
            for indices in self._buckets(candidates):
                out = self._ground(state, indices, expand=False)
                presence[indices] = out['presence_logit_dec'].sigmoid().flatten().float().cpu().numpy()
        selected = select_active_classes(presence[candidates], self.presence_threshold, self.active_keep_ratio,
                                         self.min_active_classes, self.max_active_classes)
        active = [candidates[i] for i in selected]
        self._sync()
        return active, presence, time.perf_counter() - start, math.ceil(len(candidates) / self.bucket_size)

    def _filter(self, masks, scores, labels, shape):
        from src.mask_proposer_sam3 import _nms_by_masks, _postprocess_small_regions

        valid_masks, valid_scores, valid_labels = [], [], []
        for mask, score, label in zip(masks, scores, labels):
            mask = _postprocess_small_regions(mask, self.min_mask_region_area)
            area = int(mask.sum())
            if not area or area < self.min_mask_area:
                continue
            if self.max_mask_area_ratio and area > np.prod(shape) * self.max_mask_area_ratio:
                continue
            valid_masks.append(mask)
            valid_scores.append(score)
            valid_labels.append(label)
        kept = []
        # Deduplicate within each class; do not erase another class's proposal.
        for label in sorted(set(valid_labels)):
            ids = [i for i, value in enumerate(valid_labels) if value == label]
            local = (_nms_by_masks([valid_masks[i] for i in ids], [valid_scores[i] for i in ids],
                                   self.mask_nms_threshold)
                     if self.mask_nms_threshold < 1 else range(len(ids)))
            kept.extend(ids[i] for i in local)
        kept.sort(key=lambda i: -valid_scores[i])
        return (np.stack([valid_masks[i] for i in kept]) if kept else np.zeros((0, *shape), dtype=bool),
                np.asarray([valid_scores[i] for i in kept], dtype=np.float32),
                np.asarray([valid_labels[i] for i in kept], dtype=np.int32))

    @torch.inference_mode()
    def __call__(self, img_bgr, class_indices=None):
        from src.mask_proposer_sam3 import _Instances, _Predictions

        # Release previous full-resolution arrays before processing the next image.
        self.last_result = None
        image = Image.fromarray(np.ascontiguousarray(img_bgr[..., ::-1]))
        shape = (image.height, image.width)
        candidates = (list(range(len(self.classes))) if class_indices is None
                      else list(dict.fromkeys(int(i) for i in class_indices)))
        if any(i < 0 or i >= len(self.classes) for i in candidates):
            raise ValueError('class_indices contains an out-of-range class')
        active, presence, preview_seconds, preview_calls = self._preview(image, candidates)
        self._sync()
        start = time.perf_counter()
        state = self.processor.set_image(image) if active else None
        masks, scores, labels = [], [], []
        for indices in self._buckets(active):
            out = self._ground(state, indices, expand=True)
            presence_full = out['presence_logit_dec'].sigmoid().reshape(len(indices))
            object_probs = out['pred_logits'].sigmoid().squeeze(-1) * presence_full[:, None]
            for slot, class_index in enumerate(indices):
                selected = torch.where(object_probs[slot] > self.confidence_threshold)[0]
                # Upsample a bounded number of instance masks at a time.
                for chunk in selected.split(16):
                    if not chunk.numel():
                        continue
                    probabilities = F.interpolate(out['pred_masks'][slot, chunk, None].float(),
                                                   size=shape, mode='bilinear', align_corners=False).sigmoid()[:, 0]
                    conf = object_probs[slot, chunk]
                    masks.extend((probabilities > 0.5).cpu().numpy())
                    scores.extend(conf.float().cpu().tolist())
                    labels.extend([class_index + 1] * len(chunk))
            del out
        del state
        final = self._filter(masks, scores, labels, shape)
        before_cap = len(final[0])
        final = tuple(array[:self.max_masks] for array in final) if self.max_masks else final
        self._sync()
        full_seconds = time.perf_counter() - start
        selection = []
        ranks = ({candidates[k]: rank for rank, k in enumerate(np.argsort(-presence[candidates], kind='stable'))}
                 if presence is not None else {})
        rank_count = max(self.min_active_classes, math.ceil(len(candidates) * self.active_keep_ratio))
        for i, item in enumerate(self.classes):
            if i not in candidates:
                reason = 'not_in_allowed_labels'
            elif presence is None:
                reason = 'preview_disabled' if not self.preview else 'small_candidate_set'
            elif presence[candidates].max() <= 0:
                reason = 'zero_presence_fallback' if i in active else 'zero_presence'
            else:
                candidate = presence[i] >= self.presence_threshold or ranks[i] < rank_count
                reason = ('presence_threshold' if presence[i] >= self.presence_threshold else 'rank_floor') if i in active else (
                    'active_cap' if candidate else 'below_threshold_and_rank')
            selection.append(dict(class_id=i + 1, name=item.name, retained=i in active, reason=reason))
        self.last_result = dict(
            class_ids=final[2],
            report=dict(
                classes=[dict(id=i + 1, **self.expansion[i]) for i in range(len(self.classes))],
                active_class_ids=[i + 1 for i in active],
                active_class_names=[self.classes[i].name for i in active],
                candidate_class_ids=[i + 1 for i in candidates],
                preview_presence=[float(p) if np.isfinite(p) else None for p in presence] if presence is not None else None,
                selection=selection,
                preview_seconds=preview_seconds, decode_and_filter_seconds=full_seconds,
                preview_forward_calls=preview_calls, full_forward_calls=math.ceil(len(active) / self.bucket_size),
                vocabulary_size=len(self.classes), before_cap_count=before_cap, final_count=len(final[0])))
        # The demo saves on CPU immediately; avoid copying large mask stacks back to CUDA.
        return _Predictions(_Instances(torch.from_numpy(final[0]), torch.from_numpy(final[1])))
