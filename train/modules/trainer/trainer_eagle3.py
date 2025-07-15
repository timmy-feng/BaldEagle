from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import numpy as np

from transformers import Trainer
from .trainer_utils import get_linear_schedule_with_warmup_and_decay, top_accuracy


@dataclass
class EagleMetrics3:
    classification_loss_total: torch.Tensor
    classification_loss_ttt: torch.Tensor
    top_1_acc_sum: torch.Tensor
    top_2_acc_sum: torch.Tensor
    top_3_acc_sum: torch.Tensor
    logging_valid_token_count: torch.Tensor


class EagleTrainer3(Trainer):
    def __init__(self, min_lr_ratio=0.0, ttt_length=7, discount_factor=0.8, **kwargs):
        super().__init__(**kwargs)
        self.eagle_metrics: Optional[EagleMetrics3] = None
        self.steps_since_last_logging = 0
        self.min_lr_ratio = min_lr_ratio
        self.ttt_length = ttt_length
        self.discount_factor = discount_factor

    def _ensure_metrics_on_device(self):
        if self.eagle_metrics is None:
            device = next(self.model.parameters()).device
            self.eagle_metrics = EagleMetrics3(
                classification_loss_total=torch.tensor(0.0, device=device),
                classification_loss_ttt=torch.zeros(self.ttt_length, device=device),
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
                target_head = self.model.lm_head(target_hidden_states)
                target_max_token = target_head.argmax(-1)
                target_mask = self.model.t2d[target_max_token][..., None].int()
                loss_mask = target_mask * loss_mask
                target_head = target_head[..., self.model.t2d]
                target_probs = nn.Softmax(dim=2)(target_head).detach()

                losses = []
                for i in range(self.ttt_length):
                    outputs = model(
                        input_ids=input_ids,
                        hidden_state=hidden_states,
                        attention_mask=attention_mask,
                    )

                    logits = outputs.logits
                    pred_log_probs = nn.LogSoftmax(dim=2)(logits).detach()
                    loss_class = target_probs[:, i:] * pred_log_probs
                    loss_class = -torch.sum(torch.sum(loss_mask[:, i:] * loss_class, 2)) / (
                        loss_mask[:, i:].sum() + 1e-5
                    )
                    losses.append(loss_class)

                    # save first logits for metrics
                    if i == 0:
                        pred_head = logits

                    # roll tensors for next decode
                    hidden_states = outputs.pre_norm_hidden_states[:, :-1]
                    input_ids = input_ids[:, 1:]
                    attention_mask = attention_mask[:, 1:]

                total_loss = sum(
                    np.pow(self.discount_factor, i) * loss for i, loss in enumerate(losses)
                ) / self.args.gradient_accumulation_steps

                if prediction_loss_only:
                    return (total_loss, None, None, None)
                return (total_loss, losses, pred_head, (target_head, loss_mask))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.no_grad():
                target_head = self.model.lm_head(target_hidden_states)
                target_max_token = target_head.argmax(-1)
                target_mask = self.model.t2d[target_max_token][..., None].int()
                loss_mask = target_mask * loss_mask
                target_head = target_head[..., self.model.t2d]
                target_probs = nn.Softmax(dim=2)(target_head).detach()

            losses = []
            for i in range(self.ttt_length):
                output = model(
                    input_ids=input_ids,
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                )

                logits = output.logits
                pred_log_probs = nn.LogSoftmax(dim=2)(logits)
                loss_class = target_probs[:, i:] * pred_log_probs
                loss_class = -torch.sum(torch.sum(loss_mask[:, i:] * loss_class, 2)) / (
                    loss_mask[:, i:].sum() + 1e-5
                )
                losses.append(loss_class)

                # save first logits for metrics
                if i == 0:
                    pred_head = output.logits

                # roll tensors for next decode
                hidden_states = output.pre_norm_hidden_states[:, :-1]
                input_ids = input_ids[:, 1:]
                attention_mask = attention_mask[:, 1:]

            total_loss = sum(
                np.pow(self.discount_factor, i) * loss for i, loss in enumerate(losses)
            ) / self.args.gradient_accumulation_steps

        self._ensure_metrics_on_device()

        with torch.no_grad():
            _, target = torch.max(target_head, 2)
            pred_head = pred_head.view(-1, target_head.shape[-1])
            target = target.view(-1)
            topkacc = top_accuracy(pred_head, target, (1, 2, 3), mask=(loss_mask.view(-1) == 1))

            valid_tokens = loss_mask.sum()
            assert self.eagle_metrics is not None
            self.eagle_metrics.classification_loss_total += total_loss
            self.eagle_metrics.classification_loss_ttt += torch.tensor(losses, device=losses[0].device)
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
            classification_loss = self.eagle_metrics.classification_loss_total / self.steps_since_last_logging
            classification_loss_ttt = self.eagle_metrics.classification_loss_ttt / self.steps_since_last_logging
            top_1_acc = self.eagle_metrics.top_1_acc_sum / self.eagle_metrics.logging_valid_token_count
            top_2_acc = self.eagle_metrics.top_2_acc_sum / self.eagle_metrics.logging_valid_token_count
            top_3_acc = self.eagle_metrics.top_3_acc_sum / self.eagle_metrics.logging_valid_token_count

            metrics = {
                "classification_loss": classification_loss.item(),
                "top_1_acc": top_1_acc.item(),
                "top_2_acc": top_2_acc.item(),
                "top_3_acc": top_3_acc.item(),
            }
            metrics.update({
                f"classification_loss_ttt_{i}": loss.item()
                for i, loss in enumerate(classification_loss_ttt)
            })

            self.log(metrics)

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
        eval_metrics = EagleMetrics3(
            classification_loss_total=torch.tensor(0.0, device=device),
            classification_loss_ttt=torch.zeros(self.ttt_length, device=device),
            top_1_acc_sum=torch.tensor(0.0, device=device),
            top_2_acc_sum=torch.tensor(0.0, device=device),
            top_3_acc_sum=torch.tensor(0.0, device=device),
            logging_valid_token_count=torch.tensor(0.0, device=device),
        )
        total_rows = 0

        with torch.no_grad():
            for batch in eval_dataloader:
                loss, losses, logits, labels = self.prediction_step(
                    self.model,
                    batch,
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                )
                eval_metrics.classification_loss_total += loss.detach()
                eval_metrics.classification_loss_ttt += torch.tensor(losses)

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
            "eval_loss": (eval_metrics.classification_loss_total / total_rows).item(),
            "eval_top_1_acc": (eval_metrics.top_1_acc_sum / eval_metrics.logging_valid_token_count).item(),
            "eval_top_2_acc": (eval_metrics.top_2_acc_sum / eval_metrics.logging_valid_token_count).item(),
            "eval_top_3_acc": (eval_metrics.top_3_acc_sum / eval_metrics.logging_valid_token_count).item(),
        }
        metrics.update({
            f"eval_loss_ttt_{i}": loss.item() / total_rows
            for i, loss in enumerate(eval_metrics.classification_loss_ttt)
        })

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics
