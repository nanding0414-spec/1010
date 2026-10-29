"""PRR: HCR followed by strict R3 and loose E on original-image features.

This module does not train, select thresholds, read GT or filter query features.
The HCR baseline is provided by the normal semantic rasterizer.
"""
import numpy as np


def recover_from_queries(base, masks, queries, image_id):
    from tools.r3_postfilter5_e_core import compose
    return compose(base, masks, queries, image_id)


class ProgressiveRetrievalRecovery:
    def __init__(self, checkpoint, device, extractor, gallery, labels, index, context, mapping):
        from tools import oracle_guided_r as worker
        from tools.proposal_demo.active_proposer import SAM3ActiveProposer
        model, device = worker._build_model({'sam3_weights': str(checkpoint)}, {'device': str(device)})
        self.proposer = SAM3ActiveProposer(model, device, worker._classes({}),
            preview=False, bucket_size=4, neighbor_tokens=2, hypernym_tokens=2,
            context_strength=1., force_presence_one=False, confidence_threshold=.15,
            mask_probability_threshold=.5)
        self.extractor, self.gallery, self.labels = extractor, gallery, labels
        self.index, self.context, self.mapping = index, context, mapping

    def recover(self, image_rgb, base, image_id):
        from tools.oracle_guided_r import decode_image
        from tools.r3_crossview_core import query_fields
        from src.retrieval import classify_queries
        candidates, geometry = decode_image(self.proposer, image_rgb, image_id, 'R3')
        masks = candidates['masks']
        # Mapped masks use ORIGINAL-image tokens and the SAME MLP/bank as HCR.
        vectors = self.extractor.extract_features_batch(image_rgb, list(masks)) if len(masks) else []
        if len(vectors) != len(masks):
            raise ValueError('Incomplete PRR feature extraction')
        x = np.asarray(vectors, np.float32).reshape(-1, self.gallery.shape[1])
        if not np.isfinite(x).all() or (len(x) and (np.linalg.norm(x, axis=1) <= 1e-6).any()):
            raise ValueError('Invalid PRR query; silent query removal is prohibited')
        x = np.ascontiguousarray(x/(np.linalg.norm(x, axis=1, keepdims=True)+1e-8), np.float32)
        rows = classify_queries(x, self.gallery, self.labels, self.index, self.context,
                               25, 'balanced', 5, 2, .2) if len(x) else []
        inverse = {name: int(cid) for cid, name in self.mapping.items()}
        view = dict(class_ids=np.asarray([inverse[r.label] for r in rows], np.int16),
                    retrieval_scores=np.asarray([r.score for r in rows], np.float32),
                    margins=np.asarray([r.margin for r in rows], np.float32))
        q = query_fields(view, candidates['scores'], candidates['candidate_ids'], candidates['source_classes'])
        fields, reports = recover_from_queries(base, masks, q, image_id)
        return fields['E'], dict(geometry=geometry, stages=reports,
                                original_image_semantics=True, background_only=True)
