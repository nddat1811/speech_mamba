#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "$REPO_DIR" == /kaggle/input/* ]]; then
  WORK_REPO="${WORK_REPO:-/kaggle/working/speech_mamba}"
  case "$WORK_REPO" in
    /kaggle/working/*) ;;
    *)
      echo "WORK_REPO must be inside /kaggle/working, got: $WORK_REPO" >&2
      exit 1
      ;;
  esac
  rm -rf "$WORK_REPO"
  mkdir -p "$WORK_REPO"
  cp -a "$REPO_DIR"/. "$WORK_REPO"/
  cd "$WORK_REPO"
else
  cd "$REPO_DIR"
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ $# -gt 0 ]]; then
  export CONFIG_IN="$1"
else
  export CONFIG_IN="${CONFIG_IN:-configs/spmamba-librimix.yml}"
fi
if [[ ! -f "$CONFIG_IN" ]]; then
  echo "Config file not found: $CONFIG_IN" >&2
  echo "Usage: bash scripts/kaggle_train_librimix.sh [path/to/config.yml]" >&2
  exit 1
fi
export CONFIG_OUT="${CONFIG_OUT:-/kaggle/working/spmamba-librimix-kaggle.yml}"
export DATA_ROOT="${DATA_ROOT:-/kaggle/input/librimix/DataPreProcess/Libri2Mix}"
export TRAIN_DIR="${TRAIN_DIR:-$DATA_ROOT/train-100}"
export VALID_DIR="${VALID_DIR:-$DATA_ROOT/dev}"
export TEST_DIR="${TEST_DIR:-$DATA_ROOT/test}"

python - <<PY
import os
import yaml
import torch

config_in = os.environ["CONFIG_IN"]
config_out = os.environ["CONFIG_OUT"]

with open(config_in) as f:
    conf = yaml.safe_load(f)

data = conf["datamodule"]["data_config"]
data["train_dir"] = os.environ["TRAIN_DIR"]
data["valid_dir"] = os.environ["VALID_DIR"]
data["test_dir"] = os.environ["TEST_DIR"]

if "BATCH_SIZE" in os.environ:
    data["batch_size"] = int(os.environ["BATCH_SIZE"])
if "NUM_WORKERS" in os.environ:
    data["num_workers"] = int(os.environ["NUM_WORKERS"])
else:
    data["num_workers"] = min(4, os.cpu_count() or 2)
if "EPOCHS" in os.environ:
    conf["training"]["epochs"] = int(os.environ["EPOCHS"])

gpu_count = torch.cuda.device_count()
if "GPU_IDS" in os.environ:
    conf["training"]["gpus"] = [
        int(x.strip()) for x in os.environ["GPU_IDS"].split(",") if x.strip()
    ]
elif gpu_count > 0:
    conf["training"]["gpus"] = list(range(gpu_count))
else:
    conf["training"]["gpus"] = None

os.makedirs(os.path.dirname(config_out), exist_ok=True)
with open(config_out, "w") as f:
    yaml.safe_dump(conf, f, sort_keys=False)

print("Wrote Kaggle runtime config:", config_out)
print("train_dir:", data["train_dir"])
print("valid_dir:", data["valid_dir"])
print("test_dir:", data["test_dir"])
print("gpus:", conf["training"]["gpus"])
print("epochs:", conf["training"]["epochs"])
print("batch_size:", data["batch_size"])
print("num_workers:", data["num_workers"])
PY

python audio_train.py --conf_dir="$CONFIG_OUT"
