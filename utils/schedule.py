from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import math


def get_cosine_annealing_schedule_with_warmup(optimizer: Optimizer, eta_max: float, eta_min: float, num_warmup_steps: int,
                                              num_training_steps: int, last_epoch: int = -1, num_cycles: float = 0.5):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress) +
                                   eta_min / eta_max * progress))

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def init_schedule(optimizer, args, t_total):
    if args.schedule == 'cos':
        schedule = CosineAnnealingLR(optimizer, eta_min=args.target_learning_rate, T_max=t_total)
    elif args.schedule == 'cos_w':
        schedule = get_cosine_annealing_schedule_with_warmup(optimizer, eta_max=args.learning_rate,
                                                             eta_min=args.target_learning_rate,
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=t_total)
    elif args.schedule == 'linear':
        schedule = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                   num_training_steps=t_total)
    elif args.schedule == 'one_cycle':
        schedule = OneCycleLR(optimizer,
                              max_lr=args.max_learning_rate,
                              epochs=args.num_epochs,
                              steps_per_epoch=t_total // args.num_epochs,
                              pct_start=0.2,
                              div_factor=args.max_learning_rate/args.learning_rate,
                              final_div_factor=1000)
    else:
        schedule = None
    return schedule


# ── DeepSpeed-compatible schedulers (DeepSpeed optimizer isn't torch.optim.Optimizer) ──

class DeepSpeedCosineScheduler:
    """Cosine annealing LR scheduler compatible with DeepSpeed ZeRO optimizers.

    Mimics ``CosineAnnealingLR(eta_min=min_lr, T_max=total_steps)``.
    """
    def __init__(self, optimizer, min_lr, total_steps, warmup_steps=0, base_lr=None):
        self.optimizer = optimizer
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self._step_count = 0

    def step(self):
        self._step_count += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def state_dict(self):
        return {'_step_count': self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict['_step_count']


class DeepSpeedLinearScheduler:
    """Linear warmup + linear decay scheduler for DeepSpeed."""

    def __init__(self, optimizer, warmup_steps, total_steps, base_lr=None):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr or optimizer.param_groups[0]['lr']
        self._step_count = 0

    def step(self):
        self._step_count += 1
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def _get_lr(self):
        step = self._step_count
        if step < self.warmup_steps:
            return self.base_lr * step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.base_lr * max(0.0, 1.0 - progress)

    def state_dict(self):
        return {'_step_count': self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict['_step_count']


def init_deepspeed_schedule(optimizer, args, t_total):
    """Create a scheduler that works with DeepSpeed's ZeRO optimizer."""
    if args.schedule == 'cos':
        return DeepSpeedCosineScheduler(
            optimizer, min_lr=args.target_learning_rate, total_steps=t_total,
            base_lr=args.learning_rate)
    elif args.schedule == 'cos_w':
        return DeepSpeedCosineScheduler(
            optimizer, min_lr=args.target_learning_rate, total_steps=t_total,
            warmup_steps=args.warmup_steps, base_lr=args.learning_rate)
    elif args.schedule == 'linear':
        return DeepSpeedLinearScheduler(
            optimizer, warmup_steps=args.warmup_steps, total_steps=t_total,
            base_lr=args.learning_rate)
    else:
        return None
