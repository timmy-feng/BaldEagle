import argparse
import os
import re
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=100)
parser.add_argument("--index", type=int, default=1)
parser.add_argument("--gpu_index", type=int, nargs="+", default=[0])
parser.add_argument("--outdir", type=str, default="outdir0")
parser.add_argument(
    "--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct"
)  # "meta-llama/Meta-Llama-3.1-8B-Instruct"
parser.add_argument(
    "--dataset",
    type=str,
    choices=["sharegpt", "ultrachat", "mixture_of_thoughts"],
    default="sharegpt",
)
parser.add_argument("--chat_template", type=str, default="llama")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

MAX_TOKEN_LENGTH = 2048

# ------------------------ 1. Dataset ------------------------
# This step converts the dataset into a standard messages format

if args.dataset == "sharegpt":
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
elif args.dataset == "ultrachat":
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
elif args.dataset == "mixture_of_thoughts":
    dataset = load_dataset("open-r1/Mixture-of-Thoughts", "all", split="all")

dataset = dataset.select(range(args.start, args.end))
dataset = dataset.shuffle(seed=42)

# System message that will be prepended to all conversations
system_message = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant.",
}


def format_conversation_sharegpt(row, dataset_column="conversations"):
    messages = [system_message]
    current_role = None
    for message in row[dataset_column]:
        if message["from"] == "human":
            messages.append({"role": "user", "content": message["value"]})
        elif message["from"] == "gpt":
            messages.append({"role": "assistant", "content": message["value"]})
        else:
            raise ValueError(f"Unknown role: {message['from']}")

        if current_role is None:
            current_role = messages[-1]["role"]
        else:
            assert (
                current_role != messages[-1]["role"]
            ), "Conversation has incorrect role order"
            current_role = messages[-1]["role"]

    return {"messages": messages}


def format_conversation_ultrachat(row, dataset_column="messages"):
    messages = [system_message]
    for message in row[dataset_column]:
        messages.append(message)
    return {"messages": messages}


if args.dataset == "sharegpt":
    dataset = dataset.map(format_conversation_sharegpt)
elif args.dataset == "ultrachat":
    dataset = dataset.map(format_conversation_ultrachat)
elif args.dataset == "mixture_of_thoughts":
    pass  # no need to format


# ------------------------ 2. Tokenizer ------------------------
# This step tokenizes the conversation and creates the loss mask
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Special token sequences used to identify different parts of the conversation
if args.chat_template == "llama":
    assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
elif args.chat_template == "qwen":
    assistant_header = "<|im_start|>assistant\n"
    user_header = "<|im_start|>user\n"
else:
    raise ValueError(f"Invalid chat template: {args.chat_template}")


def tokenize_conversation(row, tokenizer, col="messages"):
    formatted_conversation = tokenizer.apply_chat_template(
        row[col], tokenize=False, add_generation_prompt=False
    )

    encoding = tokenizer(
        formatted_conversation, return_offsets_mapping=True, max_length=MAX_TOKEN_LENGTH
    )
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    # Find spans of assistant responses using regex
    assistant_pattern = (
        re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
    )
    for match in re.finditer(assistant_pattern, formatted_conversation, re.DOTALL):
        # Assistant response text span (excluding assistant_header itself)
        assistant_start_char = match.start(1)
        assistant_end_char = match.end(1)

        # Mark tokens overlapping with assistant response
        for idx, (token_start, token_end) in enumerate(offsets):
            # Token is part of the assistant response span
            if token_end <= assistant_start_char:
                continue  # token before assistant text
            if token_start > assistant_end_char:
                continue  # token after assistant text
            loss_mask[idx] = 1

    return {
        "conversation_str": formatted_conversation,
        "input_ids": input_ids,
        "loss_mask": loss_mask,
    }


dataset = dataset.map(tokenize_conversation, fn_kwargs={"tokenizer": tokenizer})
dataset = dataset.remove_columns(
    [
        col
        for col in dataset.column_names
        if col not in ["input_ids", "loss_mask", "conversation_str"]
    ]
)
dataset.set_format(type="torch")

# ------------------------ 3. Compute hidden states ------------------------

model = AutoModelForCausalLM.from_pretrained(
    args.model_name, device_map="cuda", torch_dtype=torch.bfloat16
)
model.eval()

outdir = f"{args.outdir}/{args.index}"
if not os.path.exists(outdir):
    os.makedirs(outdir)

for idx, row in tqdm(enumerate(dataset), total=len(dataset)):
    # group into 5k rows per folder for huggingface upload compatibility
    group_size = 5000
    start = (idx // group_size) * group_size
    end = start + group_size
    grouped_subdir = f"rows_{start}-{end}"
    if not os.path.exists(f"{outdir}/{grouped_subdir}"):
        os.makedirs(f"{outdir}/{grouped_subdir}")

    output_file = f"{outdir}/{grouped_subdir}/data_{idx}.ckpt"
    if os.path.exists(output_file):
        continue
    with torch.no_grad():
        outputs = model(row["input_ids"].unsqueeze(0).cuda(), output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].cpu()
    data_point = {
        "input_ids": row["input_ids"],
        "loss_mask": row["loss_mask"],
        "hidden_state": hidden_states,
    }
    torch.save(data_point, output_file)
