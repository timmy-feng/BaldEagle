import modal

app = modal.App(name="eagle-generate-data")

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("eagle-data", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install([
        "accelerate",
        "datasets",
        "torch",
        "tqdm",
        "transformers",
    ])
    .add_local_file("generate_data.py", remote_path="/root/generate_data.py")
    .add_local_dir(".data", remote_path="/mnt/conversations")
)

def split_range(start, end, n, over=False):
    length = end - start + 1  # Include the end
    base_interval = length // n
    additional = length % n  # Get the remainder of the division
    intervals = []
    previous = start

    for i in range(n):
        current_interval = base_interval + (1 if i < additional else 0)
        if over:
            intervals.append((previous, previous + current_interval))
        else:
            intervals.append(
                (previous, previous + current_interval - 1)
            )  # '-1' because the end is inclusive
        previous += current_interval

    return intervals

@app.function(
    gpu='H100',
    cpu=16,
    image=image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/mnt/data": data_vol},
)
def generate_data(start: int, end: int, index: int, outdir: str, model_name: str, dataset: str, layers: str):
    import subprocess
    args = {
        "start": start,
        "end": end,
        "index": index,
        "gpu_index": 0,
        "outdir": outdir,
        "model_name": model_name,
        "dataset": dataset,
        "capture_layers": layers,
    }
    cmd = "python3 generate_data.py " + \
        " ".join([f"--{k} {v}" for k, v in args.items()])
    print(cmd)
    subprocess.run(cmd, shell=True)

@app.local_entrypoint()
def main(outdir: str, model_name: str, dataset: str, num_p: int, layers: str = "-1"):
    s = 0
    if dataset == "sharegpt":
        e = 68000 - 1
    elif dataset == "ultrachat":
        e = 200000 - 1
    elif dataset == "mixture_of_thoughts":
        e = 100000 - 1
    e = 200000 - 1

    print(f"Generating data for {dataset} from {s} to {e} in {outdir}")

    data = split_range(s, e, num_p, over=True)
    
    for _ in generate_data.starmap(
        [(start, end, index) for index, (start, end) in enumerate(data)],
        kwargs={
            "outdir": outdir,
            "model_name": model_name,
            "dataset": dataset,
            "layers": layers,
        },
    ):
        pass

    print("Done")