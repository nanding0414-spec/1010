"""Single-phase hard-center GMP, the original-image mainline aggregator."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F


def select_centers(mask, resolution=1008, grid=(72, 72)):
    m = np.asarray(mask)
    if (m.ndim != 2 or min(m.shape) < 1 or not np.isfinite(m).all()
            or not np.isin(m, [0, 1]).all()):
        raise ValueError('Expected a finite binary original-resolution mask')
    if resolution != 1008 or tuple(grid) != (72, 72):
        raise ValueError('Mainline GMP requires resolution 1008 and stride-14 72x72 tokens')
    yy, xx = np.meshgrid(np.arange(7, resolution, 14), np.arange(7, resolution, 14), indexing='ij')
    ix = np.floor(xx * m.shape[1] / resolution).astype(int).clip(0, m.shape[1]-1)
    iy = np.floor(yy * m.shape[0] / resolution).astype(int).clip(0, m.shape[0]-1)
    selected = np.flatnonzero(m[iy, ix].astype(bool).ravel())
    fallback = False
    if not len(selected) and m.any():
        ys, xs = np.nonzero(m)
        cx = (xs.mean()+.5)*resolution/m.shape[1]-.5
        cy = (ys.mean()+.5)*resolution/m.shape[0]-.5
        selected = np.asarray([int(((xx-cx)**2+(yy-cy)**2).argmin())])
        fallback = True
    return selected, dict(support=len(selected), nearest_fallback=fallback)


def uniform_gmp(tokens, indices, rho=.1):
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError('GMP rho must be finite and positive')
    if tokens.ndim != 2 or tokens.shape[1] < 1 or not tokens.is_floating_point() or not torch.isfinite(tokens).all():
        raise ValueError('Expected finite floating [N,C] tokens')
    index = torch.as_tensor(indices, dtype=torch.long, device=tokens.device)
    if index.ndim != 1 or (len(index) and (index.min() < 0 or index.max() >= len(tokens))):
        raise ValueError('Invalid center indices')
    if not len(index):
        return tokens.new_zeros(tokens.shape[1], dtype=torch.float32)
    f = tokens[index].float()
    # Identical FP32 operation order to the verified uniform-input GMP experiment.
    m = torch.ones(len(f), device=f.device, dtype=torch.float32)
    p = m / m.sum()
    energy = (p * f.square().sum(1)).sum()
    if float(energy) == 0:
        return tokens.new_zeros(tokens.shape[1], dtype=torch.float32)
    ridge = rho * energy
    mu = p @ f
    a = p.sqrt()[:, None] * f
    primal = f.shape[1] <= len(f)
    matrix = a.T @ a if primal else a @ a.T
    matrix.diagonal().add_(ridge)
    chol, flag = torch.linalg.cholesky_ex(matrix)
    if int(flag) != 0:
        raise ValueError('GMP Cholesky failed; no precision or aggregation fallback')
    rhs = mu if primal else p.sqrt()
    x = torch.cholesky_solve(rhs[:, None], chol).squeeze(1)
    z = x if primal else a.T @ x
    residual = (f.T @ (p * (f @ z)) + ridge*z - mu).norm() / mu.norm().clamp_min(1e-12)
    if not torch.isfinite(z).all() or not torch.isfinite(residual) or float(residual) > 1e-4:
        raise ValueError(f'GMP normal-equation relative residual: {float(residual)}')
    return F.normalize(z, dim=0)


def aggregate(tokens, mask, resolution=1008, grid=(72,72), rho=.1):
    if len(tokens) != grid[0]*grid[1]:
        raise ValueError('Token grid differs from mask geometry')
    indices, _ = select_centers(mask, resolution, grid)
    return uniform_gmp(tokens, indices, rho)


def describe(rho=.1):
    if not math.isfinite(rho) or rho <= 0:
        raise ValueError('GMP rho must be finite and positive')
    return dict(type='hard_center_gmp', version=1, rho=float(rho), phases=[[0,0]],
        geometry='stride14_center7_floor_original_coordinates_nearest_centroid_rowmajor',
        weights='uniform', solver='fp32_cholesky_smaller_primal_or_dual',
        residual_limit=1e-4, normalization='phase_l2_then_single_phase_mean_then_l2', empty='zero', failure='raise_no_fallback')
