import re
import glob
import h5py
import json
import os
import torch
import wandb
import warnings
import random

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from transformers.models.llama.configuration_llama import LlamaConfig

from transformers import AutoTokenizer, TrainingArguments

from modules.model.llama_eagle import LlamaForCausalLMEagle
from modules.data.data import CustomDataset, DataCollatorWithPadding, AddUniformNoise, list_files
from modules.trainer.trainer import EagleTrainer

wandb.init(project="BaldEagle")
wandb_run_name = wandb.run.name

path = "models/llama-8b/"

# -------------------------------- Load original Llama weights --------------------------------

with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    emb_path = index_json["weight_map"]["model.embed_tokens.weight"]
    lm_head_path = index_json["weight_map"]["lm_head.weight"]

with safe_open(os.path.join(path, emb_path), framework="pt", device="cpu") as f:
    tensor_slice = f.get_slice("model.embed_tokens.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim]

with safe_open(os.path.join(path, lm_head_path), framework="pt", device="cpu") as f:
    lm_head_weights = f.get_slice("lm_head.weight")[:, :]


# -------------------------------- Create draft model + tokenizer + head --------------------------------

tokenizer = AutoTokenizer.from_pretrained(path)
tokenizer.pad_token = tokenizer.eos_token

model_args = LlamaConfig(vocab_size=vocab_size,
                         hidden_size=hidden_dim,
                         intermediate_size=14336,
                         num_hidden_layers=1,
                         bos_token_id=128000,
                         eos_token_id=[128001, 128008, 128009],
                         num_key_value_heads=8,
                         num_attention_heads=32,
                         tie_word_embeddings=False,)

draft_model = LlamaForCausalLMEagle(model_args)
draft_model.load_embedding_weights(tensor)
draft_model.to("cuda:0")
draft_model.embed_tokens.weight.requires_grad = False

# Load head
head = torch.nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)
with open(os.path.join(path, "model.safetensors.index.json"), "r") as f:
    index_json = json.loads(f.read())
    head_path = index_json["weight_map"]["lm_head.weight"]
with safe_open(os.path.join(path, head_path),
                framework="pt",
                device="cpu") as f:
    tensor_slice = f.get_slice("lm_head.weight")
    vocab_size, hidden_dim = tensor_slice.get_shape()
    tensor = tensor_slice[:, :hidden_dim].float()

head.weight.data = tensor
head.to("cuda:0")
head.eval()

# -------------------------------- Load data --------------------------------

sharegpt_datapaths = list_files("/mnt/ssd4tb/sharegpt_grouped_5k/")
ultra_chat_datapaths = list_files("/mnt/ssd4tb/ultrachat_0_199999_mufp16/")

combined_data_paths = sharegpt_datapaths[:int(len(sharegpt_datapaths) * 0.95)] + ultra_chat_datapaths
random.Random(42).shuffle(combined_data_paths)
eval_data_paths = sharegpt_datapaths[int(len(sharegpt_datapaths) * 0.95):][:100]

eagle_train_dataset = CustomDataset(combined_data_paths, transform=AddUniformNoise(std=0.5))
eagle_test_dataset = CustomDataset(eval_data_paths)

eagle_collator = DataCollatorWithPadding()

# -------------------------------- Train --------------------------------

training_args = TrainingArguments(
    output_dir=f"./hf_trainer_output_dir/{wandb_run_name}/",

    num_train_epochs=10,
    gradient_accumulation_steps=16,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,
    bf16=True,
    fp16=False,
    dataloader_num_workers=4,
    
    warmup_ratio=0.01,
    learning_rate=1e-4, # 1e-3
    lr_scheduler_type="constant",  # Placeholder, we override it in the trainer

    max_grad_norm=0.5, # 1
    adam_beta1=0.9, # 0.9
    adam_beta2=0.95, # 0.999
    weight_decay=1e-2,

    eval_strategy="steps",
    logging_steps=32,
    eval_steps=64,

    save_strategy="steps",
    save_steps=0.1, # saves every 10% of training
    save_total_limit=3,
)

trainer = EagleTrainer(
    model=draft_model,
    head=head,
    args=training_args,
    train_dataset=eagle_train_dataset,
    eval_dataset=eagle_test_dataset,
    data_collator=eagle_collator,

    min_lr_ratio=0.5,  # Custmer lr scheduler param
)

trainer.train()