###
# Append epoch train/val summaries to a text log; track current best checkpoint name.
###
import os
import time
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


def _metric_scalar(trainer: pl.Trainer, *keys: str) -> float:
    metrics = trainer.callback_metrics
    for k in keys:
        if k not in metrics:
            continue
        v = metrics[k]
        if v is None:
            return float("nan")
        return float(v.item()) if hasattr(v, "item") else float(v)
    return float("nan")


class CheckpointTrainingLogCallback(pl.Callback):
    """Writes training.log (per-epoch train/val lines) and best_ckpt.log (current best ckpt stem)."""

    def __init__(
        self,
        log_dir: str,
        log_filename: str = "training.log",
        best_filename: str = "best_ckpt.log",
    ) -> None:
        super().__init__()
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, log_filename)
        self.best_log_path = os.path.join(log_dir, best_filename)
        self._train_t0: Optional[float] = None
        self._val_t0: Optional[float] = None
        self._header_written = False

    def _ensure_header(self) -> None:
        if self._header_written:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("===== Training Log =====\n")
        self._header_written = True

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        self._train_t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        self._ensure_header()
        elapsed = (
            time.perf_counter() - self._train_t0
            if self._train_t0 is not None
            else 0.0
        )
        epoch = trainer.current_epoch
        train_loss = _metric_scalar(trainer, "train_loss")
        line = (
            f"Train Summary | End of Epoch {epoch} | Time {elapsed:.3f}s | "
            f"Train Loss {train_loss:.4f}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

    def on_validation_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero:
            return
        self._val_t0 = time.perf_counter()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        elapsed = (
            time.perf_counter() - self._val_t0 if self._val_t0 is not None else 0.0
        )
        epoch = trainer.current_epoch
        val_loss = _metric_scalar(
            trainer, "val_loss/dataloader_idx_0", "val_loss"
        )
        line = (
            f"Val Summary | End of Epoch {epoch} | Time {elapsed:.3f}s | "
            f"Val Loss {val_loss:.4f}\n"
        )
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line)

        ckpt_cb = None
        for cb in trainer.callbacks:
            if isinstance(cb, ModelCheckpoint):
                ckpt_cb = cb
                break
        if ckpt_cb and getattr(ckpt_cb, "best_model_path", None):
            path = ckpt_cb.best_model_path
            if path:
                stem = Path(path).stem
                with open(self.best_log_path, "w", encoding="utf-8") as f:
                    f.write(f"{stem}\n")
