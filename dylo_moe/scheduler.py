from torch.optim.optimizer import Optimizer
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LambdaLR

def two_phase_lr_lambda(current_step: int, high_lr: float, low_lr: float, seeding_steps: int, consolidation_steps: int):
    """
    Lambda function for the two-phase learning rate scheduler.
    """
    if current_step < seeding_steps:
        return high_lr
    elif current_step < seeding_steps + consolidation_steps:
        return low_lr
    else:
        return low_lr

class TwoPhaseLRScheduler(LambdaLR):
    """
    Implements a two-phase learning rate scheduler with a high-rate seeding phase
    followed by a low-rate consolidation phase.
    """
    def __init__(self, optimizer: Optimizer, high_lr: float, low_lr: float, seeding_steps: int, consolidation_steps: int, last_epoch: int = -1):
        lr_lambda = lambda current_step: two_phase_lr_lambda(
            current_step, high_lr, low_lr, seeding_steps, consolidation_steps
        )
        super(TwoPhaseLRScheduler, self).__init__(optimizer, lr_lambda, last_epoch)