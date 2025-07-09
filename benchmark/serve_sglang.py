import modal
import modal.experimental
import os
import threading
import requests
import time

TARGET_MODEL = os.getenv("TARGET_MODEL") or "unsloth/Llama-3.1-8B-Instruct"
DRAFT_MODEL = os.getenv("DRAFT_MODEL") or "NickL77/BaldEagle-Llama-3.1-8B-Instruct"
USE_EAGLE = os.getenv("USE_EAGLE") or "1"
PORT = 8000

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .env({
        "TARGET_MODEL": TARGET_MODEL,
        "DRAFT_MODEL": DRAFT_MODEL,
        "USE_EAGLE": USE_EAGLE,
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("eagle-ckpt", create_if_missing=True)

app = modal.App(name="serve-sglang")

@app.cls(
    image=sglang_image,
    gpu="H100",
    min_containers=1,
    experimental_options={"flash": True},
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/ckpt": ckpt_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ServeSGLang():
    @modal.enter()
    def enter(self):
        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()

        # wait for port
        while True:
            try:
                requests.get(f"http://localhost:{PORT}/health")
                break
            except requests.exceptions.RequestException:
                time.sleep(1)

        self.flash_manager = modal.experimental.flash_forward(PORT)

    @modal.exit()
    def exit(self):
        self.flash_manager.stop()


def serve():
    import subprocess

    params = {
        "model": TARGET_MODEL,
        "host": "0.0.0.0",
        "dtype": "bfloat16",
        "port": PORT,
        "mem-fraction-static": 0.7,
    }

    if USE_EAGLE == "1":
        params.update({
            "speculative-algo": "EAGLE",
            "speculative-draft": DRAFT_MODEL,
            "speculative-num-steps": 5,
            "speculative-eagle-topk": 8,
            "speculative-num-draft-tokens": 64,
        })

    cmd = "python -m sglang.launch_server " + \
        " ".join([f"--{k}={v}" for k, v in params.items()])

    print(cmd)

    subprocess.Popen(cmd, shell=True)
