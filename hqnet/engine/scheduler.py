# import torch
# import math


# def build_scheduler(cfg, optimizer):
#
#     cfg_cp = cfg.scheduler.copy()
#     cfg_type = cfg_cp.pop('type')
#
#     if cfg_type not in dir(torch.optim.lr_scheduler):
#         raise ValueError("{} is not defined.".format(cfg_type))
#
#     _scheduler = getattr(torch.optim.lr_scheduler, cfg_type)
#
#     return _scheduler(optimizer, **cfg_cp)

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


class WarmupCosineAnnealingLR():
    def __init__(self, optimizer, warmup_steps, warmup_factor, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.T_max = T_max
        self.eta_min = eta_min

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                alpha = float(current_step) / float(max(1, warmup_steps))
                return warmup_factor * (1.0 - alpha) + alpha
            else:
                return 1.0

        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda)
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        self.current_step = 0

    def step(self):
        if self.current_step < self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.current_step += 1

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            return self.warmup_scheduler.get_lr()
        else:
            return self.cosine_scheduler.get_lr()


def build_scheduler(cfg, optimizer):
    cfg_cp = cfg.scheduler.copy()
    cfg_type = cfg_cp.pop('type')

    if cfg_type == 'WarmupCosineAnnealingLR':
        warmup_steps = cfg_cp.pop('warmup_steps')
        warmup_factor = cfg_cp.pop('warmup_factor')
        T_max = cfg_cp.pop('T_max')
        eta_min = cfg_cp.pop('eta_min', 0)
        return WarmupCosineAnnealingLR(optimizer, warmup_steps, warmup_factor, T_max, eta_min)

    if cfg_type not in dir(torch.optim.lr_scheduler):
        raise ValueError("{} is not defined.".format(cfg_type))

    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type)
    return _scheduler(optimizer, **cfg_cp)
