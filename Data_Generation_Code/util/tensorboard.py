# Lint as: python3
import os
from datetime import datetime
from typing import Optional, Mapping, Any, Sequence

import matplotlib.pyplot as plt
import torch
import torch.utils.tensorboard as tensorboard


class TensorBoard(object):

    def __init__(self):
        self.log_dir = None
        self.writer = None
        self.step = 0
        self.hparams = {}

    @property
    def initialized(self):
        return bool(self.log_dir)


_root = TensorBoard()


def init(log_dir: Optional[str] = None,
         comment: Optional[str] = '') -> None:
    """Creates a TensorBoard logger that will write events to the event file.

    Args:
        log_dir: Save directory location. Default is
          runs/**CURRENT_DATETIME**.
        comment: Comment log_dir suffix appended to the default
          ``log_dir``. If ``log_dir`` is assigned, this argument has no effect.
    """
    if not log_dir:
        current_time = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        log_dir = os.path.join('runs', current_time + '_' + comment)
    if not _root.initialized:
        _root.log_dir = log_dir
        _root.writer = tensorboard.SummaryWriter(log_dir=log_dir)


def get_log_dir():
    if not _root.initialized:
        init()
    return _root.log_dir


def close():
    if _root.initialized:
        _root.writer.close()


def update_step(step: int):
    if not _root.initialized:
        init()
    _root.step = step


def log_scalars(step: Optional[int] = None, write_hparams: bool = False,
                **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    for k, v in kwargs.items():
        _root.writer.add_scalar(k, v, step)
    if write_hparams:
        _root.writer.add_hparams(_root.hparams, kwargs, run_name='hparams')


def log_historgrams(step: Optional[int] = None, **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    for k, v in kwargs.items():
        _root.writer.add_histogram(k, v, step)


def log_text(step: Optional[int] = None, **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    for tag, text in kwargs.items():
        _root.writer.add_text(tag, text, step)


def log_images(step: Optional[int], tag: str, images: torch.Tensor, **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    _root.writer.add_images(tag, images, step, **kwargs)


def log_figure(step: Optional[int], **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    for tag, figure in kwargs.items():
        _root.writer.add_figure(tag, figure, step)


def register_hyper_parameter(key: str, value: Any):
    if not _root.initialized:
        init()
    _root.hparams[key] = value


def log_hyper_parameters(hyper_param: Optional[Mapping[str, Any]] = None):
    def get_writeable_type(value: Any):
        valid_types = {bool, int, float, torch.Tensor, str}
        if type(value) in valid_types:
            return value
        elif issubclass(type(value), (torch.nn.modules.loss._Loss,
                                      torch.optim.Optimizer)):
            return type(value).__name__
        else:
            return str(value)

    if not _root.initialized:
        init()
    if hyper_param is not None:
        hyper_param = {k: get_writeable_type(v) for k, v in hyper_param.items()}
        _root.hparams.update(hyper_param)


def log_graph(step: Optional[int], tag: str, x_axis: torch.Tensor,
              data: Sequence[torch.Tensor], **kwargs):
    if not _root.initialized:
        init()
    step = step if step else _root.step
    figure = plt.figure()
    legend = kwargs.pop('legend') if 'legend' in kwargs else None
    axvline = kwargs.pop('axvline') if 'axvline' in kwargs else None
    for line in data:
        plt.plot(x_axis, line, **kwargs)
    if legend:
        plt.legend(legend)
    if axvline:
        plt.axvline(x_axis[axvline], color='red', linestyle='dashed')
    _root.writer.add_figure(tag, figure, step)


def log_model(model: torch.nn.Module, input_to_model):
    """Add graph data to summary.

    Args:
        model (torch.nn.Module): Model to draw.
        input_to_model (torch.Tensor or list of torch.Tensor): A variable or a
         tuple of variables to be fed.
    """
    if not _root.initialized:
        init()
    _root.writer.add_graph(model, input_to_model)
