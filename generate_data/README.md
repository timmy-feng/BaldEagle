## Data Generation
A cleaner implementation of data generation scripts from the [official EAGLE repository](https://github.com/SafeAILab/EAGLE/tree/9367a94543e928091facd84082dcfd83ea55ca57/eagle/ge_data).

Files:
- `generate_data.py` - modularizes data generation into 3 steps:
    1. Dataset formatting
    2. Tokenization + Loss Mask Creation
    3. Hidden State Generation
    - Each step is more readable and easy to modify for different datasets and models
- `allocation.py` - a minimally modified distribued launching script copied from EAGLE
- `view_data.py` - a script to view the loss mask on a single datapoint that is generated from `generate_data.py`
    - Masked out text is in red, and valid text is in green
    - Usage: `python view_data.py --data-path {path_to_data_n.ckpt} --tokenizer {tokenizer}`

### Next Steps
- [ ] Use sglang/vllm for faster inference (hidden state generation)
