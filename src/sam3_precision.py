"""Instance-local fallback for SAM3 ViT MLP's hard-coded BF16 fused path."""
from types import MethodType


def _standard_mlp_forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.drop1(x)
    x = self.norm(x)
    x = self.fc2(x)
    return self.drop2(x)


def use_standard_vit_mlp(model):
    """Preserve weights/operations while respecting FP32 or caller autocast dtype.

    Bundled SAM3 vitdet.Mlp uses addmm_act, which casts unconditionally to
    BF16 even with autocast disabled. Patch only this model's ViT MLP instances.
    """
    count = 0
    for module in model.modules():
        cls = type(module)
        if cls.__name__ == 'Mlp' and cls.__module__ == 'sam3.model.vitdet':
            module.forward = MethodType(_standard_mlp_forward, module)
            count += 1
    return count
