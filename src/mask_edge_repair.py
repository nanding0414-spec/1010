"""Per-class binary edge repair in original image coordinates, using NumPy."""
import numpy as np


def _arguments(mask, kernel_size, iterations):
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError('Expected a finite nonempty HxW mask')
    if type(kernel_size) is not int or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer')
    if type(iterations) is not int or iterations < 0:
        raise ValueError('iterations must be a nonnegative integer')
    return value.astype(bool, copy=True)


def binary_dilate(mask, kernel_size=3, iterations=1):
    out = _arguments(mask, kernel_size, iterations)
    pad = kernel_size // 2
    for _ in range(iterations):
        padded = np.pad(out, pad, mode='constant', constant_values=False)
        result = np.zeros_like(out)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                result |= padded[dy:dy+out.shape[0], dx:dx+out.shape[1]]
        out = result
    return out


def binary_erode(mask, kernel_size=3, iterations=1):
    out = _arguments(mask, kernel_size, iterations)
    pad = kernel_size // 2
    for _ in range(iterations):
        padded = np.pad(out, pad, mode='constant', constant_values=True)
        result = np.ones_like(out)
        for dy in range(kernel_size):
            for dx in range(kernel_size):
                result &= padded[dy:dy+out.shape[0], dx:dx+out.shape[1]]
        out = result
    return out


def binary_close(mask, kernel_size=3, iterations=1):
    return binary_erode(binary_dilate(mask, kernel_size, iterations), kernel_size, iterations)


def edge_postprocess_mask(mask, close_kernel=3, close_iterations=1, dilate_pixels=1):
    out = binary_close(mask, close_kernel, close_iterations)
    return binary_dilate(out, 3, dilate_pixels).astype(np.uint8)


def sam3_logits_to_repaired_mask(mask_logits, mask_logit_threshold=.4,
                                 close_kernel=3, close_iterations=1, dilate_pixels=1,
                                 *, input_is_probability=True):
    """Threshold is always a probability; explicitly distinguish raw logits."""
    values = np.asarray(mask_logits)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('Invalid mask logits/probabilities')
    if not np.isfinite(mask_logit_threshold) or not 0 < mask_logit_threshold < 1:
        raise ValueError('Probability threshold must be between zero and one')
    if input_is_probability:
        if ((values < 0) | (values > 1)).any():
            raise ValueError('Probabilities must be in [0,1]')
        mask = values > mask_logit_threshold
    else:
        # Equivalent to sigmoid(values)>threshold, without sigmoid overflow.
        mask = values > np.log(mask_logit_threshold / (1 - mask_logit_threshold))
    return edge_postprocess_mask(mask, close_kernel, close_iterations, dilate_pixels)
