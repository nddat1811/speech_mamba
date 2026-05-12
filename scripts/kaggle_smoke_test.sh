#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python - <<'PY'
import torch
import yaml
import look2hear.models

with open("configs/spmamba-mini.yml") as f:
    conf = yaml.safe_load(f)

model = getattr(look2hear.models, conf["audionet"]["audionet_name"])(
    sample_rate=conf["datamodule"]["data_config"]["sample_rate"],
    **conf["audionet"]["audionet_config"],
)

x = torch.randn(1, int(conf["datamodule"]["data_config"]["sample_rate"]))
with torch.no_grad():
    y = model(x)

print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
print("smoke test: ok")
PY
