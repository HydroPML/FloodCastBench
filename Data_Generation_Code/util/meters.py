import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def average(self):
        return self.sum / self.count if self.count > 0 else 0


class DistributedAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, local_rank):
        self.device = local_rank
        self.reset()

    def reset(self):
        self.value = torch.zeros(1, device=self.device)
        self.sum = self.value
        self.count = 0
        self.average_value = self.value

    def update(self, val, n=1):
        self.value = torch.tensor(val, device=self.device)
        self.sum += self.value * n
        self.count += n
        self.average_value = self.sum / self.count

    @property
    def average(self):
        return self.average_value.item()

    @property
    def val(self):
        return self.value.item()
