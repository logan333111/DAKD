"""
D3KD Ablation Experiment Logger Callback.

Saves structured JSON results, per-epoch trajectory logs, and a summary CSV.

Usage (Hydra CLI):
    python src/train.py ... \
        +callbacks.ablation_logger._target_=src.callbacks.ablation_logger.AblationLogger \
        +callbacks.ablation_logger.experiment_name=exp1_tsd_metric \
        +callbacks.ablation_logger.variant_name=cls_only
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from lightning import Callback, LightningModule, Trainer


class AblationLogger(Callback):
    """Structured data logger for ablation experiments."""

    def __init__(
        self,
        experiment_name: str = "exp1_tsd_metric",
        variant_name: str = "full_tsd",
        save_dir: str = "experiments/ablation",
    ):
        super().__init__()
        self.experiment_name = experiment_name
        self.variant_name = variant_name
        self.save_dir = save_dir
        self.trajectory = []

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Record per-epoch trajectory data."""
        if not hasattr(pl_module, 'ddk_enabled') or not pl_module.ddk_enabled:
            return
        if not trainer.is_global_zero:
            return

        entry = {
            "epoch": trainer.current_epoch,
            "weights": pl_module.domain_weights.cpu().tolist(),
        }

        if hasattr(pl_module, '_ablation_log_cache'):
            entry.update(pl_module._ablation_log_cache)

        self.trajectory.append(entry)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Save structured results after testing."""
        if not trainer.is_global_zero:
            return

        if hasattr(pl_module, '_ablation_test_results'):
            results = pl_module._ablation_test_results
        else:
            results = {
                "overall_acc": trainer.callback_metrics.get("test/acc", torch.tensor(0.0)).item() * 100,
                "domain_accs": [],
                "domain_std": 0.0,
                "best_epoch": trainer.callback_metrics.get("val/acc_best", torch.tensor(0.0)).item(),
            }

        record = {
            "experiment": self.experiment_name,
            "variant": self.variant_name,
            "dataset": self._get_dataset_name(trainer),
            "config": self._extract_config(pl_module),
            "results": results,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_dir": str(trainer.default_root_dir) if trainer.default_root_dir else "",
        }

        # Save per-experiment JSON
        result_dir = os.path.join(self.save_dir, "results", self.experiment_name)
        os.makedirs(result_dir, exist_ok=True)
        json_path = os.path.join(result_dir, f"{self.variant_name}.json")
        with open(json_path, "w") as f:
            json.dump(record, f, indent=4, ensure_ascii=False)
        print(f"\n[AblationLogger] Results saved to: {json_path}")

        # Save trajectory log
        if self.trajectory:
            traj_dir = os.path.join(self.save_dir, "results", "exp5_trajectory")
            os.makedirs(traj_dir, exist_ok=True)
            traj_path = os.path.join(traj_dir, f"{self.variant_name}_trajectory.jsonl")
            with open(traj_path, "w") as f:
                for entry in self.trajectory:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(f"[AblationLogger] Trajectory saved to: {traj_path}")

        self._append_summary_csv(record)

    def _get_dataset_name(self, trainer: Trainer) -> str:
        """Extract dataset name from datamodule."""
        if hasattr(trainer, 'datamodule') and hasattr(trainer.datamodule, 'hparams'):
            return trainer.datamodule.hparams.get("data_name", "unknown")
        return "unknown"

    def _extract_config(self, pl_module: LightningModule) -> dict:
        """Extract key hyperparameters."""
        config = {}
        if hasattr(pl_module, 'hparams'):
            hp = pl_module.hparams
            config["tsd_alpha"] = getattr(hp, "tsd_alpha", None)
            config["num_domains"] = getattr(hp, "num_domains", None)
            config["ddk_enabled"] = getattr(hp, "ddk_enabled", None)
        if hasattr(pl_module, 'ablation_mode'):
            config["ablation_mode"] = pl_module.ablation_mode
        return config

    def _append_summary_csv(self, record: dict):
        """Append a row to the summary CSV."""
        csv_path = os.path.join(self.save_dir, "summary.csv")
        file_exists = os.path.exists(csv_path)

        results = record.get("results", {})
        domain_accs = results.get("domain_accs", [])

        row = {
            "experiment": record["experiment"],
            "variant": record["variant"],
            "dataset": record["dataset"],
            "overall_acc": results.get("overall_acc", ""),
            "domain_std": results.get("domain_std", ""),
            "d00_acc": domain_accs[0] if len(domain_accs) > 0 else "",
            "d09_acc": domain_accs[-1] if len(domain_accs) > 0 else "",
            "best_epoch": results.get("best_epoch", ""),
            "run_dir": record.get("run_dir", ""),
            "timestamp": record["timestamp"],
        }

        fieldnames = list(row.keys())
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"[AblationLogger] Summary appended to: {csv_path}")
