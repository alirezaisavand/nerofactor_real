# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F


def log10(x):
    """
    Computes the base-10 logarithm.
    """
    # You could simply call torch.log10(x) but we mimic the TF implementation:
    num = torch.log(x)
    denom = torch.log(torch.tensor(10.0, dtype=num.dtype, device=x.device))
    return num / denom


# -------------------------------------------------------------------
# Custom gradient functions
# -------------------------------------------------------------------

class SafeAtan2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, eps=1e-6):
        """
        Forward pass for a numerically stable atan2.
        """
        ctx.eps = eps
        ctx.save_for_backward(x, y)
        return torch.atan2(x, y)

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        eps = ctx.eps
        # Avoid division by zero by adding a small epsilon
        denom = x ** 2 + y ** 2 + eps
        dzdx = y / denom
        dzdy = -x / denom
        # Return gradients with respect to x and y; no gradient for eps.
        return grad_output * dzdx, grad_output * dzdy, None


def safe_atan2(x, y, eps=1e-6):
    """
    Numerically stable version of atan2 that avoids NaN gradients for (0,0) input.
    """
    return SafeAtan2Function.apply(x, y, eps)


class SafeAcosFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps=1e-6):
        """
        Forward pass for a numerically stable acos.
        Clips the input to [-1, 1] to avoid issues at the boundaries.
        """
        x_clip = x.clamp(-1.0, 1.0)
        ctx.eps = eps
        ctx.save_for_backward(x_clip)
        return torch.acos(x_clip)

    @staticmethod
    def backward(ctx, grad_output):
        (x_clip,) = ctx.saved_tensors
        eps = ctx.eps
        # Compute the derivative with a safeguard to avoid division by zero.
        in_sqrt = 1.0 - x_clip ** 2 + eps
        denom = torch.sqrt(in_sqrt) + eps
        dydx = -1.0 / denom
        return grad_output * dydx, None


def safe_acos(x, eps=1e-6):
    """
    Numerically stable version of acos that avoids infinite gradients at ±1.
    """
    return SafeAcosFunction.apply(x, eps)


def safe_l2_normalize(x, axis=None, eps=1e-6):
    """
    Safely L2-normalizes the tensor `x` along the specified axis.

    If `axis` is None, the entire tensor is normalized.
    """
    if axis is None:
        # Flatten x and normalize over the entire tensor.
        norm = x.norm(p=2) + eps
        return x / norm
    else:
        return F.normalize(x, p=2, dim=axis, eps=eps)


def safe_cumprod(x, eps=1e-6):
    """
    Computes an exclusive cumulative product along the last dimension.
    That is, for an input tensor x, returns a tensor y such that:
      y[..., 0] = 1
      y[..., i] = ∏_{j=0}^{i-1} (x[..., j] + eps)
    """
    x = x + eps
    # Create a tensor of ones to serve as the exclusive first element.
    ones = torch.ones_like(x[..., :1])
    cumprod = torch.cumprod(x, dim=-1)
    # Prepend ones and remove the last element to make the cumprod exclusive.
    exclusive = torch.cat([ones, cumprod[..., :-1]], dim=-1)
    return exclusive


def inv_transform_sample(val, weights, n_samples, det=False, eps=1e-5):
    """
    Inverse transform sampling.

    Given sorted sample values `val` (shape: (..., N)) and corresponding
    weights (shape: (..., N)), this function computes new samples by
    inverting the cumulative distribution function defined by `weights`.

    Arguments:
      val: Tensor of shape (..., N).
      weights: Tensor of shape (..., N).
      n_samples: Number of samples to generate.
      det: If True, use deterministic uniform samples (linspace).
           Otherwise, use random uniform samples.
      eps: A small epsilon to avoid division-by-zero.

    Returns:
      samples: Tensor of shape (..., n_samples).
    """
    # Compute the PDF and CDF.
    denom = torch.sum(weights, dim=-1, keepdim=True) + eps
    pdf = weights / denom
    cdf = torch.cumsum(pdf, dim=-1)
    # Prepend a zero to the CDF along the last dimension.
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

    # Build u: uniform samples in [0, 1] of shape (..., n_samples)
    u_shape = cdf.shape[:-1] + (n_samples,)
    if det:
        # Create a deterministic linspace.
        u = torch.linspace(0.0, 1.0, steps=n_samples, device=cdf.device)
        # Reshape and expand u to match the batch shape.
        shape_ones = (1,) * (len(cdf.shape) - 1) + (n_samples,)
        u = u.view(shape_ones).expand(u_shape)
    else:
        u = torch.rand(u_shape, device=cdf.device)

    # Find indices such that cdf[..., i-1] <= u < cdf[..., i]
    # torch.searchsorted expects cdf to be sorted along the last dimension.
    ind = torch.searchsorted(cdf, u, right=True)
    below = (ind - 1).clamp(min=0)
    above = ind.clamp(max=cdf.size(-1) - 1)
    # Stack to get indices of shape (..., n_samples, 2)
    ind_g = torch.stack([below, above], dim=-1)

    # Gather the corresponding cdf values.
    # Expand cdf to shape (..., 1, N+1) so that gather works properly.
    cdf_exp = cdf.unsqueeze(-2).expand(*cdf.shape[:-1], u.shape[-1], cdf.size(-1))
    cdf_g = torch.gather(cdf_exp, dim=-1, index=ind_g)

    # For `val` (of shape (..., N)), we need indices in the valid range [0, N-1].
    # Clamp the indices for `val` accordingly.
    ind_val = torch.stack([below, above.clamp(max=val.size(-1) - 1)], dim=-1)
    val_exp = val.unsqueeze(-2).expand(*val.shape[:-1], u.shape[-1], val.size(-1))
    val_g = torch.gather(val_exp, dim=-1, index=ind_val)

    # Compute the linear interpolation factor t within each CDF interval.
    denom_interp = cdf_g[..., 1] - cdf_g[..., 0]
    denom_interp = torch.where(denom_interp < eps, torch.ones_like(denom_interp), denom_interp)
    t = (u - cdf_g[..., 0]) / denom_interp

    # Linearly interpolate between the corresponding `val` entries.
    samples = val_g[..., 0] + t * (val_g[..., 1] - val_g[..., 0])
    return samples
