"""
D3KD Ablation Module — inherits KDModule with three independent switches:
  - tsd_enabled: TSD domain difficulty scoring (off = uniform weights)
  - ada_enabled: ADA adaptive annealing (off = fixed temperature)
  - ams_enabled: AMS asymmetric momentum smoothing (off = direct assignment)

Usage (Hydra CLI override):
    # +TSD only:
    python src/train.py ... model._target_=src.models.kd_module_ablation.KDModuleAblation \
        model.ada_enabled=false model.ams_enabled=false
    # +TSD+ADA:
    python src/train.py ... model._target_=src.models.kd_module_ablation.KDModuleAblation \
        model.ams_enabled=false
    # Full D3KD:
    python src/train.py ... model._target_=src.models.kd_module_ablation.KDModuleAblation
"""
import math
import torch
import torch.nn.functional as F
from src.models.kd_module import KDModule
import numpy as np


class KDModuleAblation(KDModule):
    """Ablation module: controls TSD/ADA/AMS via independent switches."""

    def __init__(
        self,
        # Module switches
        tsd_enabled: bool = True,
        ada_enabled: bool = True,
        ams_enabled: bool = True,
        # ADA params
        ada_mode: str = "loss_driven",    # "loss_driven" | "epoch_cosine"
        fixed_temperature: float = 1.0,
        T_max: float = 3.0,
        T_min: float = 0.2,
        target_drop: float = 0.70,
        # TSD params
        tsd_power: float = 2.0,
        # AMS params
        w_min: float = 0.65,
        w_max: float = 1.35,
        ams_eta: float = 0.2,
        beta_normal: float = 0.80,
        beta_mastered: float = 0.98,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tsd_enabled = tsd_enabled
        self.ada_enabled = ada_enabled
        self.ams_enabled = ams_enabled
        self.ada_mode = ada_mode
        self.fixed_temperature = fixed_temperature
        self.T_max = T_max
        self.T_min = T_min
        self.target_drop = target_drop
        self.tsd_power = tsd_power
        self.w_min = w_min
        self.w_max = w_max
        self.ams_eta = ams_eta
        self.beta_normal = beta_normal
        self.beta_mastered = beta_mastered

        status = (f"TSD={'on' if tsd_enabled else 'off'} | "
                  f"ADA={'on' if ada_enabled else 'off'} | "
                  f"AMS={'on' if ams_enabled else 'off'}")
        print(f"\n[AblationModule] {status}\n")

        self._ablation_test_results = {}

    def on_train_epoch_end(self) -> None:
        """Domain weight update with independent TSD/ADA/AMS switches."""
        if not self.ddk_enabled or self._local_sample_count is None:
            return

        valid_mask = self._local_sample_count > 0

        # TSD off: skip all reweighting
        if not self.tsd_enabled:
            self._log_ablation_cache(
                1.0, 0.0,
                torch.zeros(self.hparams.num_domains, device=self.device),
                torch.zeros(self.hparams.num_domains, device=self.device),
                torch.zeros(self.hparams.num_domains, device=self.device),
            )
            self._local_cls_loss_sum.zero_()
            self._local_kd_gap_sum.zero_()
            self._local_sample_count.zero_()
            return

        # TSD: domain difficulty scoring
        avg_cls_loss = torch.zeros_like(self._local_cls_loss_sum)
        avg_kd_gap = torch.zeros_like(self._local_kd_gap_sum)
        avg_cls_loss[valid_mask] = self._local_cls_loss_sum[valid_mask] / self._local_sample_count[valid_mask]
        avg_kd_gap[valid_mask] = self._local_kd_gap_sum[valid_mask] / self._local_sample_count[valid_mask]

        norm_cls, norm_kd = avg_cls_loss.clone(), avg_kd_gap.clone()
        if valid_mask.sum() > 0:
            cls_mean = avg_cls_loss[valid_mask].mean()
            kd_mean = avg_kd_gap[valid_mask].mean()
            if cls_mean > 0: norm_cls[valid_mask] /= cls_mean
            if kd_mean > 0: norm_kd[valid_mask] /= kd_mean

        base_difficulty = self.hparams.tsd_alpha * norm_cls + (1 - self.hparams.tsd_alpha) * norm_kd
        domain_difficulty = base_difficulty ** self.tsd_power

        # ADA: annealing temperature
        current_global_cls_loss = avg_cls_loss[valid_mask].mean().item()
        if getattr(self, "initial_base_loss", None) is None or self.current_epoch == 0:
            self.initial_base_loss = current_global_cls_loss

        if self.ada_enabled:
            if self.ada_mode == "epoch_cosine":
                progress = self.current_epoch / max(self.trainer.max_epochs - 1, 1)
            else:  # loss_driven (default)
                drop_ratio = (self.initial_base_loss - current_global_cls_loss) / (self.initial_base_loss + 1e-6)
                progress = max(0.0, min(drop_ratio / self.target_drop, 1.0))
            dynamic_temp = self.T_min + 0.5 * (self.T_max - self.T_min) * (1.0 + math.cos(math.pi * progress))
        else:
            dynamic_temp = self.fixed_temperature
            progress = 0.0

        # Weight computation
        new_weights = F.softmax(domain_difficulty / dynamic_temp, dim=0) * self.hparams.num_domains
        new_weights = torch.clamp(new_weights, min=self.w_min, max=self.w_max)

        # AMS: asymmetric momentum smoothing
        if self.ams_enabled:
            perfect_threshold = torch.quantile(avg_cls_loss[valid_mask], self.ams_eta).item()
            is_perfect = (avg_cls_loss < perfect_threshold) & valid_mask
            dynamic_beta = torch.full_like(self.domain_weights, self.beta_normal)
            dynamic_beta[is_perfect] = self.beta_mastered
            self.domain_weights.copy_(dynamic_beta * self.domain_weights + (1 - dynamic_beta) * new_weights)
        else:
            self.domain_weights.copy_(new_weights)

        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'update_domain_weights'):
            self.trainer.datamodule.update_domain_weights(self.domain_weights.cpu().clone())

        self._log_ablation_cache(dynamic_temp, progress, domain_difficulty, avg_cls_loss, avg_kd_gap)

        if self.trainer.is_global_zero and self.current_epoch % 10 == 0:
            status = (f"TSD={'on' if self.tsd_enabled else 'off'} "
                      f"ADA={'on' if self.ada_enabled else 'off'} "
                      f"AMS={'on' if self.ams_enabled else 'off'}")
            print(f"\n[Ablation Epoch {self.current_epoch}] {status} | "
                  f"Temp: {dynamic_temp:.3f}, Progress: {progress*100:.1f}%")
            print(f" {'Dom':^4} | {'Diff':^8} | {'Weight':^8} | {'ClsLoss':^8} | {'Count':^6}")
            for d in range(self.hparams.num_domains):
                if self._local_sample_count[d] > 0:
                    print(f" {d:02d}   | {domain_difficulty[d]:8.4f} | {self.domain_weights[d]:8.3f} "
                          f"| {avg_cls_loss[d]:8.4f} | {int(self._local_sample_count[d]):6d}")
            print()

        self._local_cls_loss_sum.zero_()
        self._local_kd_gap_sum.zero_()
        self._local_sample_count.zero_()

    def _log_ablation_cache(self, dynamic_temp, progress, domain_difficulty, avg_cls_loss, avg_kd_gap):
        """Cache epoch data for AblationLogger callback."""
        self._ablation_log_cache = {
            "temperature": dynamic_temp,
            "progress": progress,
            "difficulty": domain_difficulty.cpu().tolist(),
            "avg_cls_loss": avg_cls_loss.cpu().tolist(),
            "avg_kd_gap": avg_kd_gap.cpu().tolist(),
        }

    def on_test_epoch_end(self) -> None:
        """Override test end to cache structured results."""
        if not self.test_step_outputs:
            return

        local_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        local_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
        has_domains = "domain_ids" in self.test_step_outputs[0]

        all_preds = self.all_gather(local_preds).view(-1)
        all_targets = self.all_gather(local_targets).view(-1)

        if has_domains:
            local_domains = torch.cat([x["domain_ids"] for x in self.test_step_outputs])
            all_domains = self.all_gather(local_domains).view(-1)

        if self.trainer.is_global_zero:
            all_preds = all_preds.cpu()
            all_targets = all_targets.cpu()
            overall_acc = (all_preds == all_targets).float().mean().item() * 100

            domain_accs = []
            if has_domains:
                all_domains = all_domains.cpu()
                for d in range(self.hparams.num_domains):
                    mask = (all_domains == d)
                    if mask.sum() > 0:
                        acc = (all_preds[mask] == all_targets[mask]).float().mean().item() * 100
                        domain_accs.append(round(acc, 2))
                    else:
                        domain_accs.append(0.0)

            self._ablation_test_results = {
                "overall_acc": round(overall_acc, 2),
                "domain_accs": domain_accs,
                "domain_std": round(float(np.std(domain_accs)), 2) if domain_accs else 0.0,
                "best_epoch": self.trainer.checkpoint_callback.best_model_path if self.trainer.checkpoint_callback else "",
            }

        super().on_test_epoch_end()
