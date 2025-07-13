import argparse
from datasets import load_dataset
from tqdm import tqdm
from openai import OpenAI
import concurrent.futures
import json
import uuid
import time
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable OpenAI HTTP request logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

MAX_TOKENS = 2048
BATCH_POLL_INTERVAL = 5
ROUTING_HEADER = "X-Modal-Flash-Upstream"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate assistant turns from a dataset")
    parser.add_argument("--outfile", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--base_url", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_concurrency", type=int, default=16)
    return parser.parse_args()

def load_preset_dataset(dataset_name, start, end):
    if dataset_name == "sharegpt":
        dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    elif dataset_name == "ultrachat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    elif dataset_name == "mixture_of_thoughts":
        dataset = load_dataset("HuggingFaceH4/mixture-of-thoughts", split="train")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = dataset.select(range(start, end))
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


    if dataset_name == "sharegpt":
        dataset = dataset.map(format_conversation_sharegpt)
    elif dataset_name == "ultrachat":
        dataset = dataset.map(format_conversation_ultrachat)
    elif dataset_name == "mixture_of_thoughts":
        pass  # no need to format

    return dataset

def get_masked_conversations(batch):
    conversations = []
    skipped = 0
    for row in batch:
        messages = []
        is_invalid = False
        for i, message in enumerate(row["messages"]):
            expected_role = "system" if i == 0 else ["assistant", "user"][i % 2]
            if message["role"] != expected_role:
                is_invalid = True
                skipped += 1
                break
            # mask out assistant responses
            messages.append({
                "role": message["role"],
                "content": message["content"] if message["role"] != "assistant" else None,
            })
        if not is_invalid:
            conversations.append(messages)
    if skipped > 0:
        logger.info(f"Skipped {skipped} invalid conversations")
    return conversations

def send_batch(file_lines, openai_client, batch_id):
    batch_input_file = openai_client.files.create(
        file="\n".join(file_lines).encode(),
        purpose="batch",
        extra_headers={ROUTING_HEADER: batch_id},
    )

    batch = openai_client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        extra_headers={ROUTING_HEADER: batch_id},
    )

    while True:
        batch = openai_client.batches.retrieve(batch.id, extra_headers={ROUTING_HEADER: batch_id})
        if batch.status == "completed":
            break
        elif batch.status == "in_progress":
            time.sleep(BATCH_POLL_INTERVAL)
            continue
        else:
            raise ValueError(f"Batch {batch.id} has status {batch.status}")

    file_response = openai_client.files.content(
        batch.output_file_id,
        extra_headers={ROUTING_HEADER: batch_id},
    )

    return file_response.text

def process_batch(batch, openai_client, model_name):
    conversations = get_masked_conversations(batch)
    batch_id = str(uuid.uuid4())

    while True:
        file_lines = []
        id_to_index = {}

        for i, conversation in enumerate(conversations):
            item_id = str(uuid.uuid4())

            conversation_prefix = []
            for message in conversation:
                if message["content"] is not None:
                    conversation_prefix.append(message)
                else:
                    break

            # if entire conversation has been generated, skip
            if len(conversation_prefix) == len(conversation):
                continue

            file_lines.append(json.dumps({
                "custom_id": item_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": conversation_prefix,
                    "max_tokens": MAX_TOKENS,
                }
            }))
            id_to_index[item_id] = i

        # if no more assistant turns to generate, we're done
        if len(file_lines) == 0:
            return [{"messages": conversation} for conversation in conversations]

        while True:
            try:
                file_text = send_batch(file_lines, openai_client, batch_id=batch_id)
                break
            except Exception as e:
                logger.error(f"Error sending batch: {e}")
                time.sleep(BATCH_POLL_INTERVAL)
                continue

        for line in file_text.splitlines():
            try:
                output = json.loads(line)
                index = id_to_index[output["custom_id"]]
                completion = output["response"]["body"]["choices"][0]["message"]["content"]
                conversation = conversations[index]
                for message in conversation:
                    if message["content"] is None:
                        message["content"] = completion
                        break
            except Exception as e:
                logger.error(f"Error processing line {line}: {e}")
                continue

        logger.info(f"Extended {len(file_lines)} turns")

def main():
    args = parse_arguments()

    s = 0
    if args.dataset == "sharegpt":
        e = 68000 - 1
    elif args.dataset == "ultrachat":
        e = 200000 - 1
    elif args.dataset == "mixture_of_thoughts":
        e = 100000 - 1

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    dataset = load_preset_dataset(args.dataset, s, e)
    openai_client = OpenAI(base_url=args.base_url, api_key="NOT_USED")

    batches = [dataset.select(range(i, min(i + args.batch_size, e))) for i in range(s, e, args.batch_size)]

    logger.info(f"Generating data for {args.dataset} from {s} to {e} in {args.outfile} with {len(batches)} batches")

    f = open(args.outfile, "w")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
        futures = [executor.submit(process_batch, batch, openai_client, args.model_name) for batch in batches]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing batches"):
            for conversation in future.result():
                f.write(json.dumps(conversation) + "\n")

    f.close()

if __name__ == "__main__":
    main()
