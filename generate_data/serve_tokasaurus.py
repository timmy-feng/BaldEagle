import modal
import modal.experimental
import os
import threading
import requests
import time

MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.1-8B-Instruct"
PORT = 10210

tokasaurus_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/ScalingIntelligence/tokasaurus.git /root/tokasaurus",
    )
    .run_commands("cd /root/tokasaurus && pip install -e .")
    .run_commands("pip install --force-reinstall pydra-config==0.0.15")
    .env({
        "MODEL_NAME": MODEL_NAME,
    })
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

app = modal.App(name="serve-tokasaurus")

DP_SIZE = 8

@app.cls(
    image=tokasaurus_image,
    gpu=f"H100:{DP_SIZE}",
    min_containers=4,
    experimental_options={"flash": True},
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class ServeTokasaurus():
    @modal.enter()
    def enter(self):
        self.server_thread = threading.Thread(target=serve, daemon=True)
        self.server_thread.start()

        # wait for port
        while True:
            try:
                requests.get(f"http://localhost:{PORT}/ping")
                break
            except requests.exceptions.RequestException:
                time.sleep(1)

        self.flash_manager = modal.experimental.flash_forward(PORT)

    @modal.exit()
    def exit(self):
        self.flash_manager.stop()

def serve():
    import subprocess
    subprocess.Popen(f"toka dp_size={DP_SIZE} model={MODEL_NAME} port={PORT}", shell=True)
