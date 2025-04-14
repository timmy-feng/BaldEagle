import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from transformers import Trainer

def top_accuracy(output, target, topk=(1,)):
    # output.shape (bs, num_classes), target.shape (bs, )
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, -1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.item())
        return res

class EagleTrainer(Trainer):

    def __init__(self, head, **kwargs):
        super().__init__(**kwargs)
        self.regression_loss_total = 0
        self.classification_loss_total = 0
        self.top_1_acc_sum = 0
        self.top_2_acc_sum = 0
        self.top_3_acc_sum = 0
        self.logging_valid_token_count = 0
        self.steps_since_last_logging = 0
        self.head = head

    # def _setup_model(self, model, *args, **kwargs):
    #     # Do not wrap the model in DataParallel even if multiple GPUs are available.
    #     return model
    
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]

        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            with torch.no_grad():
                predicted_hidden_states = model(input_ids=input_ids,
                                                hidden_state=hidden_states,
                                                attention_mask=attention_mask)
                        
                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head).detach()
                _, target = torch.max(target_head, 2)

                pred_head = self.head(predicted_hidden_states)
                pred_log_probs = nn.LogSoftmax(dim=2)(pred_head).detach()
                loss_class = target_probs * pred_log_probs
                loss_class = -torch.sum(torch.sum(loss_mask * loss_class, 2)) / (loss_mask.sum() + 1e-5)

                # Compute regression loss 
                loss_reg = nn.SmoothL1Loss(reduction="none")(predicted_hidden_states, target_hidden_states)
                loss_reg = torch.sum(torch.mean(loss_mask * loss_reg, 2)) / (loss_mask.sum() + 1e-5)
                total_loss = loss_reg + (0.1 * loss_class)

                if prediction_loss_only:
                    return (total_loss, None, None)
                return (total_loss, pred_head, (target_head, loss_mask))

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):       
        input_ids = inputs["input_ids"]
        hidden_states = inputs["hidden_states"]
        attention_mask = inputs["attention_mask"]
        target_hidden_states = inputs["target"]
        loss_mask = inputs["loss_mask"][:, :, None]
        
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            # Forward pass in one go.

            predicted_hidden_states = model(input_ids=input_ids,
                                            hidden_state=hidden_states,
                                            attention_mask=attention_mask)
        
            with torch.no_grad():
                target_head = self.head(target_hidden_states)
                target_probs = nn.Softmax(dim=2)(target_head).detach()
            
            pred_head = self.head(predicted_hidden_states)
            pred_log_probs = nn.LogSoftmax(dim=2)(pred_head).detach()
            loss_class = target_probs * pred_log_probs
            loss_class = -torch.sum(torch.sum(loss_mask * loss_class, 2)) / (loss_mask.sum() + 1e-5)

            # Compute regression loss 
            loss_reg = nn.SmoothL1Loss(reduction="none")(predicted_hidden_states, target_hidden_states)
            loss_reg = torch.sum(torch.mean(loss_mask * loss_reg, 2)) / (loss_mask.sum() + 1e-5)

            total_loss = (loss_reg + (0.3 * loss_class)) / self.args.gradient_accumulation_steps

        with torch.no_grad():
            _, target = torch.max(target_head, 2)
            pred_head = pred_head.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
            target = target.view(-1)[loss_mask.view(-1) == 1]
            topkacc = top_accuracy(pred_head, target, (1, 2, 3))

            valid_tokens = loss_mask.sum().item()
            self.regression_loss_total += loss_reg.item() / self.args.gradient_accumulation_steps
            self.classification_loss_total += loss_class.item() / self.args.gradient_accumulation_steps
            # self.top_1_acc_sum += topkacc[0] / (valid_tokens + 1e-5)
            # self.top_2_acc_sum += topkacc[1] / (valid_tokens + 1e-5)
            # self.top_3_acc_sum += topkacc[2] / (valid_tokens + 1e-5)
            self.top_1_acc_sum += topkacc[0]
            self.top_2_acc_sum += topkacc[1]
            self.top_3_acc_sum += topkacc[2]
            self.logging_valid_token_count += valid_tokens

        if model.training:
            self.steps_since_last_logging += 1

        return total_loss
    
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time):

        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            self.log({
                "regression_loss": self.regression_loss_total / self.steps_since_last_logging,
                "classification_loss": self.classification_loss_total / self.steps_since_last_logging,
                "top_1_acc": self.top_1_acc_sum / self.logging_valid_token_count,
                "top_2_acc": self.top_2_acc_sum / self.logging_valid_token_count,
                "top_3_acc": self.top_3_acc_sum / self.logging_valid_token_count,
            })
            
            self.regression_loss_total = 0
            self.classification_loss_total = 0
            self.top_1_acc_sum = 0
            self.top_2_acc_sum = 0
            self.top_3_acc_sum = 0
            self.steps_since_last_logging = 0
            self.logging_valid_token_count = 0

            self.control.should_log = True
        
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)
    
    def evaluate(self, eval_dataset=None, ignore_keys=False):
        eval_dataset = eval_dataset or self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        self.model.eval()
    
        top_1_acc_sum, top_2_acc_sum, top_3_acc_sum, total_rows, total_valid_tokens = 0, 0, 0, 0, 0
        losses = []
    
        with torch.no_grad():
            for batch in eval_dataloader:
                loss, logits, labels = self.prediction_step(
                    self.model, batch, prediction_loss_only=False, ignore_keys=ignore_keys
                )
                losses.append(loss.detach().cpu())

                target_head, loss_mask = labels

                _, target = torch.max(target_head, 2)
                pred_head = logits.view(-1, target_head.shape[-1])[loss_mask.view(-1) == 1]
                target = target.view(-1)[loss_mask.view(-1) == 1]
                topkacc = top_accuracy(pred_head, target, (1, 2, 3))

                valid_tokens = loss_mask.sum().item()
                top_1_acc_sum += topkacc[0]
                top_2_acc_sum += topkacc[1]
                top_3_acc_sum += topkacc[2]
                total_valid_tokens += valid_tokens
                # TODO: this only assumes bs=1, match bs
                total_rows += 1
        
        metrics = {
            "eval_loss": np.mean(losses).item(),
            "eval_top_1_acc": top_1_acc_sum  / total_valid_tokens,
            "eval_top_2_acc": top_2_acc_sum / total_valid_tokens,
            "eval_top_3_acc": top_3_acc_sum / total_valid_tokens,
        }

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        return metrics
