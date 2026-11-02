"""Shared same-image encoding, promoted from the accepted paired VOC regression."""
from contextlib import contextmanager
import torch


@contextmanager
def use_model_for_extractor(model):
    # The normal extractor still performs its own identity/precision/geometry
    # checks and dimension probe. Only its model-construction dependency changes.
    import src.feature_extractor_sam3 as module
    old = module.build_sam3_model
    def same_model(**kwargs):
        assert not kwargs['enable_inst_interactivity']
        assert kwargs['enable_segmentation']
        return model
    module.build_sam3_model = same_model
    try:
        yield
    finally:
        module.build_sam3_model = old


class ImageEncodingReuse:
    """Capture this image's trunk/FPN values, then replay only identical inputs.

    The proposal decoder is unchanged. Preview and crop are never substituted
    for original-image features. Hooks and overrides are scoped and restored.
    """
    def __init__(self, extractor):
        self.extractor = extractor
        self.trunk = extractor.model.backbone.vision_backbone.trunk
        self.conv = extractor._select_fpn_conv(extractor.model.backbone.vision_backbone)
        self.saved = None
        self.handles = []
        self.replayed_trunk = self.replayed_fpn = 0

    @contextmanager
    def capture(self):
        if self.saved is not None or self.handles:
            raise RuntimeError('Unconsumed or nested encoding capture')
        captured = {}
        def trunk_hook(module, args, output):
            if tuple(args[0].shape[-2:]) == (1008, 1008):
                if 'trunk' in captured:
                    raise RuntimeError('More than one full original-image encoding')
                if not isinstance(output, (list, tuple)) or not all(isinstance(x, torch.Tensor) for x in output):
                    raise TypeError('Unsupported trunk output container')
                captured['input'] = args[0].detach().clone()
                captured['trunk'] = type(output)(x.detach().clone() for x in output)
        def fpn_hook(module, args, output):
            if 'trunk' in captured:
                if 'fpn' in captured:
                    raise RuntimeError('More than one selected FPN output')
                captured['fpn_input'] = args[0].detach().clone()
                captured['fpn'] = output.detach().clone()
        self.handles = [self.trunk.register_forward_hook(trunk_hook), self.conv.register_forward_hook(fpn_hook)]
        try:
            yield
            if set(captured) != {'input', 'trunk', 'fpn_input', 'fpn'}:
                raise RuntimeError('Full-resolution proposal encoding was not captured')
            self.saved = captured
        finally:
            for h in self.handles:
                h.remove()
            self.handles = []

    @torch.inference_mode()
    def tokens(self, rgb):
        if self.saved is None:
            raise RuntimeError('No encoding for this image')
        saved, self.saved = self.saved, None  # Single use, including on failure.
        inp, _ = self.extractor._preprocess_image(rgb)
        if inp.dtype != saved['input'].dtype or not torch.equal(inp, saved['input']):
            raise ValueError('Different preprocessing/image/view cannot share encoding')
        old_trunk, old_conv = self.trunk.forward, self.conv.forward
        calls = [0, 0]
        def trunk(x):
            if not torch.equal(x, saved['input']):
                raise ValueError('Trunk input differs')
            calls[0] += 1
            return saved['trunk']
        def conv(x):
            if not torch.equal(x, saved['fpn_input']):
                raise ValueError('FPN input differs')
            calls[1] += 1
            return saved['fpn']
        self.trunk.forward, self.conv.forward = trunk, conv
        try:
            result = self.extractor._forward_tokens(inp)
            if calls != [1, 1]:
                raise RuntimeError(f'Unexpected replay path: {calls}')
            self.replayed_trunk += 1
            self.replayed_fpn += 1
            return result
        finally:
            self.trunk.forward, self.conv.forward = old_trunk, old_conv
