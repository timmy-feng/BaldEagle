# Start an sglang server
# python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3.1-8B-Instruct --host 0.0.0.0

import argparse
import os
import re
import torch

from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description='sp')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=100)
parser.add_argument('--index', type=int, default=1)
parser.add_argument('--gpu_index', type=int, nargs='+', default=[0])
parser.add_argument('--outdir', type=str, default='outdir0')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)[1:-1]

# ------------------------ 1. Dataset ------------------------
# This step converts the dataset into a standard messages format

# dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

dataset = dataset.select(range(args.start, args.end))
dataset = dataset.shuffle(seed=42)

# System message that will be prepended to all conversations
system_message = {
    "role": "system",
    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
}

# def format_conversation_sharegpt(row, dataset_column="conversations"):
#     messages = [system_message]
#     current_role = None
#     for message in row[dataset_column]:
#         if message["from"] == "human":
#             messages.append({
#                 "role": "user",
#                 "content": message["value"]}
#             )
#         elif message["from"] == "gpt":
#             messages.append({
#                 "role": "assistant",
#                 "content": message["value"]}
#             )
#         else:
#             raise ValueError(f"Unknown role: {message['from']}")
        
#         if current_role is None:
#             current_role = messages[-1]["role"]
#         else:
#             assert current_role != messages[-1]["role"], f"Conversation has incorrect role order"
#             current_role = messages[-1]["role"]
    
#     return {"messages": messages}

# dataset = dataset.map(format_conversation_sharegpt)

def format_conversation_ultrachat(row, dataset_column="messages"):
    messages = [system_message]
    for message in row[dataset_column]:
        messages.append(message)
    return {"messages": messages}

dataset = dataset.map(format_conversation_ultrachat)

# ------------------------ 2. Tokenizer ------------------------
# This step tokenizes the conversation and creates the loss mask
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Special token sequences used to identify different parts of the conversation
assistant_header = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
user_header = "<|eot_id|><|start_header_id|>user<|end_header_id|>"

def tokenize_conversation(row, tokenizer, col="messages"):
    formatted_conversation = tokenizer.apply_chat_template(
        row[col],
        tokenize=False,
        add_generation_prompt=False
    )
    
    encoding = tokenizer(formatted_conversation, return_offsets_mapping=True)
    input_ids = encoding.input_ids
    offsets = encoding.offset_mapping
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

    # Find spans of assistant responses using regex
    assistant_pattern = re.escape(assistant_header) + r"(.*?)(?=" + re.escape(user_header) + "|$)"
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


    return {"conversation_str": formatted_conversation,
            "input_ids": input_ids,
            "loss_mask": loss_mask}

dataset = dataset.map(tokenize_conversation, fn_kwargs={"tokenizer": tokenizer})
dataset = dataset.remove_columns(
    [
        col for col in dataset.column_names
        if col not in ["input_ids", "loss_mask", "conversation_str"]
    ]
)
dataset.set_format(type="torch")

# ------------------------ 3. Compute hidden states ------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
model.eval()

outdir = f'{args.outdir}/{args.index}'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for idx, row in tqdm(enumerate(dataset)):
    output_file = f'{outdir}/data_{idx}.ckpt'
    if os.path.exists(output_file):
        continue
    with torch.no_grad():
        outputs = model(row["input_ids"].unsqueeze(0).cuda(), output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1].cpu()
    data_point = {
        "input_ids": row["input_ids"],
        "loss_mask": row["loss_mask"],
        "hidden_state": hidden_states
    }
    torch.save(data_point, output_file)
