"""
Copied from https://github.com/sgl-project/sglang/blob/aac531c53b0166bc3883d1f6491f7f0fbb928197/benchmark/mtbench/bench_sglang_eagle.py

Benchmark SGLang EAGLE/EAGLE3 Speculative Decoding

Usage:
python3 benchmark/mtbench/bench_sglang_eagle.py --num-questions 80 --parallel 1
"""

import argparse
import json
import time

import sglang as sgl
from sglang.test.test_utils import (
    add_common_sglang_args_and_parse,
    select_sglang_backend,
)


def load_questions(filename):
    questions = []
    with open(filename, "r") as fin:
        for line in fin:
            obj = json.loads(line)
            questions.append(obj)
    return questions


@sgl.function
def answer_mt_bench(s, question_1, question_2):
    s += sgl.system(
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    )
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1"))


def main(args):
    # Construct prompts
    questions = load_questions(args.question_file)[: args.num_questions]
    arguments = [
        {"question_1": q["turns"][0], "question_2": q["turns"][1]} for q in questions
    ]

    # Select backend - check for remote URL first
    if hasattr(args, 'remote_url') and args.remote_url:
        backend = sgl.RuntimeEndpoint(args.remote_url)
        print(f"Using remote SGLang server at: {args.remote_url}")
    else:
        backend = select_sglang_backend(args)
        print("Using local SGLang backend")
    
    sgl.set_default_backend(backend)

    # Run requests
    tic = time.time()
    rets = answer_mt_bench.run_batch(
        arguments,
        temperature=0,
        max_new_tokens=2048,
        num_threads=args.parallel,
        progress_bar=True,
    )

    latency = time.time() - tic
    num_output_tokens = sum(
        s.get_meta_info("answer_1")["completion_tokens"] for s in rets
    )

    # NOTE: acceptance length is just completion_tokens / spec_verify_ct
    # {'id': '3bb9c5ead109488d8ed5ee9cbecaec29', 'finish_reason': {'type': 'length', 'length': 256}, 'prompt_tokens': 37, 'spec_verify_ct': 101, 'completion_tokens': 256, 'cached_tokens': 0}

    output_throughput = num_output_tokens / latency

    has_verify = "spec_verify_ct" in rets[0].get_meta_info("answer_1")
    if has_verify:
        num_verify_tokens = sum(
            s.get_meta_info("answer_1")["spec_verify_ct"] for s in rets
        )

        accept_length = num_output_tokens / num_verify_tokens
    else:
        accept_length = 1.0

    print(
        f"#questions: {len(questions)}, Throughput: {output_throughput:.2f} token/s, Acceptance length: {accept_length:.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default=None)
    parser.add_argument("--num-questions", type=int, default=80)
    parser.add_argument("--remote-url", type=str, default=None, 
                        help="URL of remote SGLang server (e.g., http://localhost:30000)")
    args = add_common_sglang_args_and_parse(parser)
    main(args)
