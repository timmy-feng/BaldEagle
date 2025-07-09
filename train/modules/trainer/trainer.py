import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

from transformers import Trainer
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


@dataclass
class EagleMetrics:
    regression_loss_total: torch.Tensor
    classification_loss_total: torch.Tensor
    top_1_acc_sum: torch.Tensor
    top_2_acc_sum: torch.Tensor
    top_3_acc_sum: torch.Tensor
    logging_valid_token_count: torch.Tensor


class EagleTrainer(Trainer):
    def __init__(self, head, min_lr_ratio=0.0, **kwargs):
        super().__init__(**kwargs)
        self.eagle_metrics: Optional[EagleMetrics] = None
        self.steps_since_last_logging = 0
        self.head = head
        self.min_lr_ratio = min_lr_ratio

    def _ensure_metrics_on_device(self):
        if self.eagle_metrics is None:
            device = next(self.model.parameters()).device
            self.eagle_metrics = EagleMetrics(
                regression_loss_total=torch.tensor(0.0, device=device),
                classification_loss_total=torch.tensor(0.0, device=device),
                top_1_acc_sum=torch.tensor(0.0, device=device),
                top_2_acc_sum=torch.tensor(0.0, device=device),
                top_3_acc_sum=torch.tensor(0.0, device=device),
                logging_valid_token_count=torch.tensor(0.0, device=device),
            )

    def create_scheduler(self, num_training_steps, optimizer=None):
        """
        Override to create a custom linear decay scheduler with warmup.
        """
        if optimizer is None:
            optimizer = self.optimizer

        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)

        self.lr_scheduler = get_linear_schedule_with_warmup_and_decay(
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr_ratio=self.min_lr_ratio,
        )

        print(f"Created custom LR scheduler with min_lr_ratio={self.min_lr_ratio}")

        return self.lr_scheduler

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.no_grad():
                predicted_hidden_states = model(
                    input_ids=input_ids,
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                )

                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head).detach()
                _, target = torch.max(target_head, 2)

                pred_head = self.head(predicted_hidden_states)
                pred_log_probs = nn.LogSoftmax(dim=2)(pred_head).detach()
                loss_class = target_probs * pred_log_probs
                loss_class = -torch.sum(torch.sum(loss_mask * loss_class, 2)) / (
                    loss_mask.sum() + 1e-5
                )

                # Compute regression loss
                loss_reg = nn.SmoothL1Loss(reduction="none")(
                    predicted_hidden_states, target_hidden_states
                )
                loss_reg = torch.sum(torch.mean(loss_mask * loss_reg, 2)) / (
                    loss_mask.sum() + 1e-5
                )
                total_loss = loss_reg + (0.1 * loss_class)

                if prediction_loss_only:
                    return (total_loss, None, None)
                return (total_loss, pred_head, (target_head, loss_mask))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Forward pass in one go.

            predicted_hidden_states = model(
                input_ids=input_ids,
                hidden_state=hidden_states,
                attention_mask=attention_mask,
            )

            with torch.no_grad():
                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head).detach()

            pred_head = self.head(predicted_hidden_states)
            pred_log_probs = nn.LogSoftmax(dim=2)(pred_head).detach()
            loss_class = target_probs * pred_log_probs
            loss_class = -torch.sum(torch.sum(loss_mask * loss_class, 2)) / (
                loss_mask.sum() + 1e-5
            )

            # Compute regression loss
            loss_reg = nn.SmoothL1Loss(reduction="none")(
                predicted_hidden_states, target_hidden_states
            )
            loss_reg = torch.sum(torch.mean(loss_mask * loss_reg, 2)) / (
                loss_mask.sum() + 1e-5
            )

            total_loss = (
                loss_reg + (0.1 * loss_class)
            ) / self.args.gradient_accumulation_steps

        self._ensure_metrics_on_device()

        with torch.no_grad():
            _, target = torch.max(target_head, 2)
            pred_head = pred_head.view(-1, target_head.shape[-1])
            target = target.view(-1)
            topkacc = top_accuracy(pred_head, target, (1, 2, 3), mask=(loss_mask.view(-1) == 1))

            valid_tokens = loss_mask.sum()
            self.eagle_metrics.regression_loss_total += loss_reg
            self.eagle_metrics.classification_loss_total += loss_class
            self.eagle_metrics.top_1_acc_sum += topkacc[0]
            self.eagle_metrics.top_2_acc_sum += topkacc[1]
            self.eagle_metrics.top_3_acc_sum += topkacc[2]
            self.eagle_metrics.logging_valid_token_count += valid_tokens

        if model.training:
            self.steps_since_last_logging += 1

        return total_loss

    def _maybe_log_save_evaluate(
        self,
        tr_loss,
        grad_norm,
        model,
        trial,
        epoch,
        ignore_keys_for_eval,
        start_time,
        learning_rate=None,
    ):
        if (
            self.control.should_log
            and self.state.global_step > self._globalstep_last_logged
            and self.eagle_metrics is not None
        ):
            regression_loss = self.eagle_metrics.regression_loss_total / self.steps_since_last_logging
            classification_loss = self.eagle_metrics.classification_loss_total / self.steps_since_last_logging
            top_1_acc = self.eagle_metrics.top_1_acc_sum / self.eagle_metrics.logging_valid_token_count
            top_2_acc = self.eagle_metrics.top_2_acc_sum / self.eagle_metrics.logging_valid_token_count
            top_3_acc = self.eagle_metrics.top_3_acc_sum / self.eagle_metrics.logging_valid_token_count

            self.log(
                {
                    "regression_loss": regression_loss.item(),
                    "classification_loss": classification_loss.item(),
                    "top_1_acc": top_1_acc.item(),
                    "top_2_acc": top_2_acc.item(),
                    "top_3_acc": top_3_acc.item(),
                }
            )

            self.eagle_metrics = None
            self.steps_since_last_logging = 0
            self.control.should_log = True

        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=False):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()

        device = next(self.model.parameters()).device
        eval_metrics = EagleMetrics(
            regression_loss_total=torch.tensor(0.0, device=device),
            classification_loss_total=torch.tensor(0.0, device=device),
            top_1_acc_sum=torch.tensor(0.0, device=device),
            top_2_acc_sum=torch.tensor(0.0, device=device),
            top_3_acc_sum=torch.tensor(0.0, device=device),
            logging_valid_token_count=torch.tensor(0.0, device=device),
        )
        total_rows = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss, logits, labels = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                )
                eval_metrics.regression_loss_total += loss.detach()

                target_head, loss_mask = labels

                _, target = torch.max(target_head, 2)
                pred_head = logits.view(-1, target_head.shape[-1])
                target = target.view(-1)
                topkacc = top_accuracy(pred_head, target, (1, 2, 3), mask=(loss_mask.view(-1) == 1))

                valid_tokens = loss_mask.sum()
                eval_metrics.top_1_acc_sum += topkacc[0]
                eval_metrics.top_2_acc_sum += topkacc[1]
                eval_metrics.top_3_acc_sum += topkacc[2]
                eval_metrics.logging_valid_token_count += valid_tokens
                # TODO: this only assumes bs=1, match bs
                total_rows += 1

        metrics = {
            "eval_loss": (eval_metrics.regression_loss_total / total_rows).item(),
            "eval_top_1_acc": (eval_metrics.top_1_acc_sum / eval_metrics.logging_valid_token_count).item(),
            "eval_top_2_acc": (eval_metrics.top_2_acc_sum / eval_metrics.logging_valid_token_count).item(),
            "eval_top_3_acc": (eval_metrics.top_3_acc_sum / eval_metrics.logging_valid_token_count).item(),
        }

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics
