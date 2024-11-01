import torch
import torch.nn as nn

G = 9.8


def downsample(z: torch.Tensor, ds_factor: int) -> torch.Tensor:
    """downsample 2d tensor z by a factor of ds_factor.

    z should have 3 dimensions of (batch size, rows, cols). if z is provided
    with 2 dimensions, a third (batch size = 1) is deduced automatically.
    The returned downsampled tensor has 3 dimensions (batch size, rows, cols).
    """
    if z.dim() == 2:
        z = z.expand(1, *z.shape)
    ds_operator = nn.AvgPool2d(kernel_size=ds_factor)
    return ds_operator(z)


def cfl(dx: float, max_h: torch.Tensor, alpha: float) -> torch.Tensor:
    return (alpha * dx / (G + max_h)).reshape(-1, 1, 1, 1)


class CFL(object):
    def __init__(self, dx, alpha):
        self.dx = dx
        self.alpha = alpha

    def __call__(self, max_h):
        return (
            (self.alpha * self.dx / (G + max_h)).reshape(-1, 1, 1, 1))
