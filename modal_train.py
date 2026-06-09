"""
Run autoresearch training on Modal's serverless GPUs.

Drop-in replacement for `uv run train.py` — just uses a remote GPU instead.
Output streams in real-time so you can watch training or redirect to a log file.

Usage:
    modal run modal_train.py --prepare   # one-time data prep
    modal run modal_train.py             # run training
"""

import modal

# ---------------------------------------------------------------------------
# Image Build Functions
# ---------------------------------------------------------------------------

def download_model_weights():
    """
    Triggered during the Modal image build step. 
    Instantiating the model forces the weights to download and bake into the image.
    """
    from mosaic.models.boltz2 import Boltz2
    from mosaic.proteinmpnn.mpnn import load_mpnn_sol
    from mosaic.models.esmfold2 import ESMFold2Full
    print("Downloading Boltz2 weights into container image...")
    _ = Boltz2()
    print("Boltz2 Weights successfully cached.")
    print("Downloading MPNN Weights")
    _ = load_mpnn_sol(0.05)
    print("SolMPNN Weights successfully cached")
    print("Downloading Full ESMFold2 Model Weights with MSA Generation & Usage for Predicting Target Structures ")
    _ = ESMFold2Full()
    print("ESMFold2Full Weights have successfully been cached")


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

data_volume = modal.Volume.from_name("autoresearch-data", create_if_missing=True)
cache_volume = modal.Volume.from_name("autoresearch-cache", create_if_missing=True)

DATA_MOUNT = "/root/.cache/autoresearch"
CACHE_MOUNT = "/root/.cache/jax-compile"

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("uv") # uv must be installed first to run the following commands
    .apt_install("git")
    .run_commands("git clone https://github.com/escalante-bio/mosaic.git /mosaic")
    .workdir("/mosaic")
    .run_commands("uv pip install --system -r pyproject.toml")
    .run_commands("uv pip install --system .")
    # Install GPU JAX last so mosaic's CPU jax dependency cannot downgrade it
    .run_commands("uv pip install --system 'jax[cuda12]' equinox scikit-learn biotite")
    .run_function(download_model_weights)
    .env({
        "JAX_COMPILATION_CACHE_DIR": CACHE_MOUNT,
        # Ensure python finds both your local scripts (/app) and mosaic (/mosaic)
        "PYTHONPATH": "/app:/mosaic",
        # High memory fraction for large binder + target complex
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
    })
    .add_local_dir(".", remote_path="/app")
)

app = modal.App("hallucinate", image=image)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

@app.function(
    volumes={DATA_MOUNT: data_volume},
    timeout=600,
    cpu=4,
)
def prepare_data():
    """Extract motif coordinates and setup initial state. Run once."""
    import subprocess
    subprocess.run(
        ["python", "/app/prepare.py"],
        check=True,
    )
    data_volume.commit()


@app.function(
    gpu="H100",
    volumes={
        DATA_MOUNT: data_volume,
        CACHE_MOUNT: cache_volume,
    },
    timeout=3600, # Extended timeout for large multi-step PSSM optimization (In Seconds)
)
def train():
    """Run train.py on a remote GPU. Output streams to stdout."""
    import subprocess
    result = subprocess.run(
        ["python", "-u", "/app/train.py"],
        cwd="/app",
    )
    data_volume.commit()
    cache_volume.commit()
    if result.returncode != 0:
        raise SystemExit(result.returncode)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    prepare: bool = False,
):
    """Run hallucination pipeline on a Modal GPU.

    Usage:
        modal run modal_train.py             # run training remotely
        modal run modal_train.py --prepare   # one-time data prep
    """
    if prepare:
        print("Preparing initial motif data...")
        prepare_data.remote()
        print("Data preparation complete.")
        return

    # Use train.local() here instead if you want to test on your local machine's GPU
    train.remote()