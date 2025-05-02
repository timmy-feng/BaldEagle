## Benchmarking

This repo uses [sglang](https://github.com/sgl-project/sglang) to benchmark the performance of trained models. Sglang already implements EAGLE2 dynamic draft trees.

To benchmark:
1. Start an sglang server with EAGLE speculative decoding:
```
python3 -m sglang.launch_server \
    --model {target model} \
    --speculative-algo EAGLE \
    --speculative-draft {path to your EAGLE checkpoint} \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 8 \
    --speculative-num-draft-tokens 64 \
    --dtype bfloat16 \
    --port 30000 \
    --mem-fraction-static 0.9
```
- You can lower `mem-fraction-static` if running into OOM errors. 

2. Then in another terminal, run either `python bench_sglang_eagle_double_turn.py` (standard) or `python bench_sglang_eagle_single_turn.py` (faster)
- optionally add `--num_questions N` parameter to benchmark on a subset