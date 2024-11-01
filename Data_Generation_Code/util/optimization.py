import torch
import torch.nn as nn
import torch.nn.functional as F


class InundationLoss(nn.Module):
    def __init__(self, threshold: float = 0.5, reduction: str = 'mean'):
        super(InundationLoss, self).__init__()
        self.reduction = reduction
        self.threshold = threshold

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        difference = torch.abs(input - target)
        loss = torch.as_tensor((difference > self.threshold), dtype=torch.float)
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        return loss


def scale_regularization(output: torch.Tensor, target: torch.Tensor, criterion):
    """Max - Min regularization."""
    output_max_value = torch.max(output)
    output_min_value = torch.min(output)
    target_max_value = torch.max(target)
    target_min_value = torch.min(target)
    return criterion(output_max_value - output_min_value,
                     target_max_value - target_min_value)


def lpf_regularization(output: torch.Tensor, target: torch.Tensor, criterion):
    """Low pass filter similarity regulrization."""
    low_pass_filter = torch.ones((1, 1, 2, 2)) / 4
    low_pass_filter = low_pass_filter.to(output.device)
    filtered_output = F.conv2d(F.pad(output, (1, 0, 1, 0), mode='replicate'),
                               low_pass_filter)
    return criterion(filtered_output, target.to(output.device))
