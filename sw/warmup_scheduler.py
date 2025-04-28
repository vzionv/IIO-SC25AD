from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, warm_up_init_lr=1e-6, after_scheduler=None, expo=False):
        groups_len = len(optimizer.param_groups)
        if isinstance(warm_up_init_lr, float):
            warm_up_init_lr = [warm_up_init_lr] * groups_len
        if isinstance(warm_up_init_lr, list):
            assert len(warm_up_init_lr) == groups_len
        else:
            raise TypeError('init_lr should be a float or a list of float with len = %d' % groups_len)
        self.warm_up_init_lrs = warm_up_init_lr
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.expo = expo
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.expo:
            return [
                (init_lr - ((math.exp(self.total_epoch) * init_lr - base_lr) / (math.exp(self.total_epoch) - 1))) * math.exp(
                    self.last_epoch) + (math.exp(self.total_epoch) * init_lr - base_lr) / (math.exp(self.total_epoch) - 1) for
                (base_lr, init_lr) in zip(self.base_lrs, self.warm_up_init_lrs)]

        if self.multiplier == 1.0:
            return [(base_lr - init_lr) * (float(self.last_epoch) / self.total_epoch) + init_lr for (base_lr, init_lr) in
                    zip(self.base_lrs, self.warm_up_init_lrs)]
        else:
            return [(base_lr - init_lr) * (
                    (self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1. + init_lr * (self.multiplier - 1.)) for
                    (base_lr, init_lr) in zip(self.base_lrs, self.warm_up_init_lrs)]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
