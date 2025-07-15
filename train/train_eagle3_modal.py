import modal 

import json
import torch
import wandb
import random
from typing import Any

from safetensors import safe_open
from huggingface_hub import hf_hub_download

from transformers.models.llama.configuration_llama import LlamaConfig
import transformers

from transformers import AutoTokenizer, TrainingArguments, TrainerCallback

from modules.model import LlamaForCausalLMEagle3
from modules.data.data import (
    Eagle3LocalDataset,
    DataCollatorWithPadding,
    AddUniformNoise,
    list_local_files,
)
from modules.trainer import EagleTrainer3

app = modal.App(name="eagle-train")

flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install([
        "accelerate==1.8.1",
        "datasets",
        "numpy==2.2.4",
        "torch==2.6.0",
        "tqdm",
        "transformers==4.53.1",
        "wandb",
        flash_attn_release,
    ])
    .add_local_dir("modules", "/root/modules")
)
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("eagle-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("eagle-ckpt", create_if_missing=True)

def download_model_files(model_path: str):
    """Download all necessary model files from HF hub and return their local paths"""
    try:
        index_path = hf_hub_download(repo_id=model_path, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index_json = json.loads(f.read())
            emb_filename = index_json["weight_map"]["model.embed_tokens.weight"]
            head_filename = index_json["weight_map"]["lm_head.weight"]
    except Exception:
        print("Repo did not have a model.safetensors.index.json, assuming single safetensors file.")
        emb_filename = "model.safetensors"
        head_filename = "model.safetensors"
    
    emb_path = hf_hub_download(repo_id=model_path, filename=emb_filename)
    head_path = hf_hub_download(repo_id=model_path, filename=head_filename)

    config_path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(config_path, "r") as f:
        config_json = json.loads(f.read())
    
    return {
        "emb_path": emb_path,
        "head_path": head_path,
        "config_json": config_json,
    }

def load_original_weights(model_files: dict):
    with safe_open(model_files["emb_path"], framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice("model.embed_tokens.weight")
        vocab_size, hidden_dim = tensor_slice.get_shape()
        tensor = tensor_slice[:, :hidden_dim].float()
    return vocab_size, hidden_dim, tensor

def create_draft_model(vocab_size: int, draft_vocab_size: int, hidden_dim: int, tensor: torch.Tensor, model_files: Any):
    model_args = LlamaConfig(
        vocab_size=vocab_size,
        draft_vocab_size=draft_vocab_size,
        hidden_size=hidden_dim,
        intermediate_size=model_files["config_json"]["intermediate_size"],
        num_hidden_layers=1,
        bos_token_id=128000,
        eos_token_id=[128001, 128008, 128009],
        num_key_value_heads=8,
        num_attention_heads=32,
        tie_word_embeddings=model_files["config_json"]["tie_word_embeddings"],
    )

    draft_model = LlamaForCausalLMEagle3(model_args)
    draft_model.load_embedding_weights(tensor)
    draft_model.to("cuda")
    draft_model.embed_tokens.weight.requires_grad = False

    return draft_model

def load_data(sharegpt_path: str, ultrachat_path: str):
    sharegpt_datapaths = list_local_files(sharegpt_path)
    ultrachat_datapaths = list_local_files(ultrachat_path)
    combined_data_paths = sharegpt_datapaths[: int(len(sharegpt_datapaths) * 0.95)] + ultrachat_datapaths
    random.Random(42).shuffle(combined_data_paths)
    eval_data_paths = sharegpt_datapaths[int(len(sharegpt_datapaths) * 0.95) :][:100]

    eagle_train_dataset = Eagle3LocalDataset(
        combined_data_paths,
        transform=AddUniformNoise(std=0.5),
    )
    eagle_test_dataset = Eagle3LocalDataset(eval_data_paths)

    return eagle_train_dataset, eagle_test_dataset

NUM_DATALOADER_WORKERS = 64

@app.function(
    image=image,
    gpu='H100',
    cpu=NUM_DATALOADER_WORKERS,
    timeout=60 * 60 * 24,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("wandb-secret")
    ],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/data": data_vol,
        "/ckpt": ckpt_vol,
    },
    ephemeral_disk=1024 * 1024,
    memory=1024 * 128,
)
def train(model_path: str, sharegpt_path: str, ultrachat_path: str, outdir: str, profile: bool = False):
    wandb.init(project="BaldEagle3")
    wandb_run_name = wandb.run.name

    model_files = download_model_files(model_path)
    vocab_size, hidden_dim, tensor = load_original_weights(model_files)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    # TODO: make draft vocab size configurable
    draft_model = create_draft_model(
        vocab_size=vocab_size,
        draft_vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        tensor=tensor,
        model_files=model_files,
    )

    draft_model.d2t = torch.arange(vocab_size).to(draft_model.device)
    draft_model.t2d = torch.ones(vocab_size, dtype=torch.bool).to(draft_model.device)

    eagle_train_dataset, eagle_test_dataset = load_data(sharegpt_path, ultrachat_path)
    eagle_collator = DataCollatorWithPadding()

    transformers.logging.set_verbosity_info()
    torch._dynamo.config.capture_scalar_outputs = True

    training_args = TrainingArguments(
        output_dir=f"{outdir}/{wandb_run_name}/",
        num_train_epochs=5,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        remove_unused_columns=False,
        bf16=True,
        fp16=False,
        dataloader_num_workers=NUM_DATALOADER_WORKERS,
        dataloader_prefetch_factor=4,
        warmup_ratio=0.01,
        learning_rate=1e-4,  # 1e-3
        lr_scheduler_type="constant",  # Placeholder, we override it in the trainer
        max_grad_norm=0.5,  # 1
        adam_beta1=0.9,  # 0.9
        adam_beta2=0.95,  # 0.999
        weight_decay=1e-2,
        eval_strategy="steps",
        logging_steps=256,
        eval_steps=1024,
        dataloader_pin_memory=True,
        accelerator_config={"non_blocking": True},
        logging_nan_inf_filter=False,
        save_strategy="steps",
        save_steps=1024,
        torch_compile=True,
    )

    if profile:
        training_args.max_steps = 128

    trainer = EagleTrainer3(
        model=draft_model,
        ttt_length=7,
        args=training_args,
        train_dataset=eagle_train_dataset,
        eval_dataset=eagle_test_dataset,
        data_collator=eagle_collator,
        min_lr_ratio=0.5,  # Custmer lr scheduler param
    )

    if profile:
        trainer.add_callback(ProfileCallback(f"{outdir}/{wandb_run_name}/"))

    trainer.train()

    print("Done")

class ProfileCallback(TrainerCallback):
    def __init__(self, outdir: str, wait: int = 80, warmup: int = 4, active: int = 16, repeat: int = 1):
        self.outdir = outdir
        self.profiler = None
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        
    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize profiler at the start of training"""
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"{self.outdir}/tensorboard_traces"
            ),
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
            with_modules=True,
        )
        self.profiler.start()
        
    def on_step_end(self, args, state, control, **kwargs):
        """Step the profiler at the end of each training step"""
        if self.profiler is not None:
            self.profiler.step()
            
    def on_train_end(self, args, state, control, **kwargs):
        """Stop the profiler at the end of training"""
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None

@app.local_entrypoint()
def main(model_path: str, sharegpt_path: str, ultrachat_path: str, outdir: str, profile: bool = False):
    train.remote(
        model_path=model_path,
        sharegpt_path=sharegpt_path,
        ultrachat_path=ultrachat_path, 
        outdir=outdir,
        profile=profile,
    )
