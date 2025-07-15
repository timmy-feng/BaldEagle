import torch
from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup_and_decay(
    optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.0, last_epoch=-1
):
    """
    Create a schedule with linear warmup and then linear decay.

    Args:
        optimizer: The optimizer for which to schedule the learning rate
        num_warmup_steps: The number of steps for the warmup phase
        num_training_steps: The total number of training steps
        min_lr_ratio: The final learning rate ratio (min_lr = lr * min_lr_ratio)
        last_epoch: The index of the last epoch when resuming training

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule
    """

    def lr_lambda(current_step: int):
        # Warmup phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Linear decay phase
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def top_accuracy(output, target, topk=(1,), mask=None):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, -1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = (correct[:k].float() * mask.float()).sum()
            res.append(correct_k)
        return res
