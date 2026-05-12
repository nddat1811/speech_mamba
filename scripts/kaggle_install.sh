#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
import importlib.util
import subprocess
import sys

packages = []
if importlib.util.find_spec("causal_conv1d") is None:
    packages.append("causal-conv1d==1.2.0.post2")
if importlib.util.find_spec("mamba_ssm") is None:
    packages.append("mamba-ssm==1.2.0.post1")

if packages:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", *packages]
    )
PY

python - <<'PY'
import torch
import pytorch_lightning as pl
import mamba_ssm

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda devices:", torch.cuda.device_count())
print("pytorch_lightning:", pl.__version__)
print("mamba_ssm import: ok")
PY
