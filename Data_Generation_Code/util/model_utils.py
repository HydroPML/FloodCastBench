# Lint as: python3
import logging
from typing import Optional, Mapping, Text

import torch
import torch.nn as nn


def get_model_gradients(model: nn.Module) -> Mapping[Text, torch.Tensor]:
    gradients = {}
    for name, weight in model.named_parameters():
        gradients[name] = weight.grad.data.clone()
    return gradients


def calc_gradient_norm(model: nn.Module, p: Optional[int] = 2) -> float:
    """Calculates the gradient p-norm of the provided model."""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(p)
            total_norm += param_norm.item() ** p
        else:
            logging.warning(
                'parameter %s in model does not have gradient', name)
    return total_norm ** (1. / p)


def calc_weight_norm(model: nn.Module, p: Optional[int] = 2) -> float:
    """Calculates the weights p-norm of the provided model."""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_norm = param.norm(p)
            total_norm += param_norm.item() ** p
        else:
            logging.warning(
                'parameter %s in model does require gradient', name)
    return total_norm ** (1. / p)
