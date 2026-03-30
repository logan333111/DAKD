from typing import Any, Dict, Tuple
import torch
import torch.nn.functional as F
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import numpy as np


class KDModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_teacher: bool,
        kd_criterion,
        compile: bool,
        ddk_enabled: bool = True,       # DDK 总开关
        num_domains: int = 5,           # 域数量
        ddk_alpha: float = 0.5,         # 保留向后兼容（已废弃）
        tsd_alpha: float = 0.2,         # TSD 难度计算的平衡系数
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.net = net
        self.use_teacher = use_teacher
        if use_teacher:
            self.kd_criterion = kd_criterion

        self.ddk_enabled = ddk_enabled
        
        # [核心隔离] 根据是否开启 DDK，初始化不同的标准损失函数
        if self.ddk_enabled:
            # DDK 需要逐样本 loss 以及可能的平滑
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
            
            # DDK 专属缓存与累加器
            self.register_buffer('domain_weights', torch.ones(num_domains))
            self._local_cls_loss_sum = None
            self._local_kd_gap_sum = None
            self._local_sample_count = None
        else:
            # 严格对齐原论文基线
            self.criterion = torch.nn.CrossEntropyLoss()

        # 指标定义
        num_classes = self.hparams.net.data_attributes.class_num
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        if use_teacher:
            self.kd_loss = MeanMetric()
            self.cls_loss = MeanMetric()
            self.img_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

        if self.ddk_enabled:
            n = self.hparams.num_domains
            self._local_cls_loss_sum = torch.zeros(n, device=self.device)
            self._local_kd_gap_sum = torch.zeros(n, device=self.device)
            self._local_sample_count = torch.zeros(n, device=self.device)

    def model_step(self, batch: Tuple):
        """前向传播 + 损失计算"""
        # [还原] 严格使用原论文的全局退火策略，保证控制变量绝对公平
        def calculate_loss_weights(current_epoch, total_epochs):
            weight1 = current_epoch / (total_epochs * 8)
            weight1 = np.clip(weight1, 0, 1)
            weight2 = 1 - weight1
            weight2 = np.clip(weight2, 0, 1)
            return weight1, weight2

        # 动态解包，兼容基线 Dataset 可能只返回 (x, y) 的情况
        x = batch[0]
        y = batch[1]
        domain_ids = batch[2] if len(batch) > 2 else None

        loss_dict = {}
        cls_loss_vec = None
        per_sample_kd_gap = None

        if self.use_teacher:
            outputs = self.forward(x)
            img_loss, kd_loss = self.kd_criterion(outputs)
            
            # 基础分类损失计算
            if self.ddk_enabled:
                cls_loss_vec = self.criterion(outputs[1], y)
                base_cls_loss = cls_loss_vec.mean()
            else:
                base_cls_loss = self.criterion(outputs[1], y)
                
            cls_loss_weight, kd_loss_weight = calculate_loss_weights(self.current_epoch, self.trainer.max_epochs)

            if self.ddk_enabled:
                # =======================================================
                # DDK 专属逻辑区 (只有 ddk_enabled=True 时才执行，不浪费基线算力)
                # =======================================================
                hidden_features, out, clip_img_features, clip_nlp_features, aligned_img, aligned_nlp = outputs
                with torch.no_grad():
                    per_sample_img_gap = (hidden_features - aligned_img).abs().mean(dim=-1)
                    temp, scale = self.kd_criterion.temperature, self.kd_criterion.logit_scale
                    student_logits = scale * hidden_features @ aligned_nlp.T / temp
                    teacher_logits = scale * clip_img_features @ clip_nlp_features.T / temp
                    per_sample_kl = F.kl_div(
                        F.log_softmax(student_logits, dim=1),
                        F.softmax(teacher_logits, dim=1),
                        reduction='none'
                    ).sum(dim=1) * (temp ** 2)
                    per_sample_kd_gap = per_sample_img_gap + per_sample_kl

                # DAR 域权重加权
                sample_weights = self.domain_weights[domain_ids]
                weighted_cls_loss = (cls_loss_vec * sample_weights).mean()
                batch_avg_weight = sample_weights.mean()
                weighted_img_loss = img_loss * batch_avg_weight
                weighted_kd_loss = kd_loss * batch_avg_weight
                
                final_loss = cls_loss_weight * weighted_cls_loss + (kd_loss_weight * weighted_img_loss + kd_loss_weight * weighted_kd_loss) / 2
                
            else:
                # =======================================================
                # 严格的原版基线计算区
                # =======================================================
                cls_loss = cls_loss_weight * base_cls_loss
                weighted_img_loss = kd_loss_weight * img_loss
                weighted_kd_loss = kd_loss_weight * kd_loss
                final_loss = cls_loss + (weighted_img_loss + weighted_kd_loss) / 2

            preds = torch.argmax(outputs[1], dim=1)
            loss_dict.update({
                "loss": final_loss, 
                "cls_loss": base_cls_loss, 
                "img_loss": img_loss, 
                "kd_loss": kd_loss
            })

        else:
            logits = self.forward(x)
            if self.ddk_enabled:
                loss_vec = self.criterion(logits, y)
                cls_loss_vec = loss_vec
                sample_weights = self.domain_weights[domain_ids]
                final_loss = (loss_vec * sample_weights).mean()
            else:
                final_loss = self.criterion(logits, y)

            preds = torch.argmax(logits, dim=1)
            loss_dict["loss"] = final_loss

        return (
            final_loss, loss_dict, preds, y,
            cls_loss_vec.detach() if cls_loss_vec is not None else None,
            per_sample_kd_gap.detach() if per_sample_kd_gap is not None else None,
            domain_ids,
        )

    def training_step(self, batch, batch_idx):
        loss, loss_dicts, preds, targets, cls_loss_vec, per_sample_kd_gap, domain_ids = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        
        # 恢复使用原本的 self.log 以确保格式完全对齐
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.use_teacher:
            self.cls_loss(loss_dicts["cls_loss"])
            self.img_loss(loss_dicts["img_loss"])
            self.kd_loss(loss_dicts["kd_loss"])
            self.log("train/cls_loss", self.cls_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/img_loss", self.img_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/kd_loss", self.kd_loss, on_step=False, on_epoch=True, prog_bar=True)

        # DDK 独有的本地数据累加
        if self.ddk_enabled and self.training:
            with torch.no_grad():
                for d in range(self.hparams.num_domains):
                    mask = (domain_ids == d)
                    count = mask.sum()
                    if count > 0:
                        self._local_cls_loss_sum[d] += cls_loss_vec[mask].sum()
                        self._local_sample_count[d] += count
                        if per_sample_kd_gap is not None:
                            self._local_kd_gap_sum[d] += per_sample_kd_gap[mask].sum()

        return loss

    def on_train_epoch_end(self) -> None:
        """基于本地统计数据的域难度评估与权重更新 (仅 DDK 开启时执行)"""
        if not self.ddk_enabled or self._local_sample_count is None:
            return

        valid_mask = self._local_sample_count > 0
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

        import math

        base_difficulty = self.hparams.tsd_alpha * norm_cls + (1 - self.hparams.tsd_alpha) * norm_kd
        domain_difficulty = base_difficulty ** 2.0 

        # ============================================================
        # 🚀 自适应数据驱动退火 (完全基于 cls_loss 的真实进度)
        # ============================================================
        current_global_cls_loss = avg_cls_loss[valid_mask].mean().item()

        # 1. 记录初始分类 Loss 作为基准点
        if getattr(self, "initial_base_loss", None) is None or self.current_epoch == 0:
            self.initial_base_loss = current_global_cls_loss

        # 2. 计算纯分类 Loss 下降的比例
        drop_ratio = (self.initial_base_loss - current_global_cls_loss) / (self.initial_base_loss + 1e-6)

        # 3. 设定目标阈值：根据 TensorBoard 曲线，设定为 0.70 最稳妥
        target_drop = 0.70 
        
        # 将其映射到 0~1 的进度条
        progress = max(0.0, min(drop_ratio / target_drop, 1.0))

        # 4. 余弦退火公式 
        T_max = 3.0 
        T_min = 0.2 
        dynamic_temp = T_min + 0.5 * (T_max - T_min) * (1.0 + math.cos(math.pi * progress))
        # ============================================================

        new_weights = F.softmax(domain_difficulty / dynamic_temp, dim=0) * self.hparams.num_domains
        
       # ============================================================
        # 🛡️ 优化防掉点机制：非对称动量平滑 (Asymmetric EMA)
        # ============================================================
        new_weights = torch.clamp(new_weights, min=0.65, max=1.45)
        
        perfect_threshold = 0.05
        is_perfect = (avg_cls_loss < perfect_threshold) & valid_mask
        
        # 动态生成平滑动量 beta
        # 正常域 beta = 0.8，完美域 beta = 0.98 (极强惯性，抵御剧烈下降)
        dynamic_beta = torch.full_like(self.domain_weights, 0.8)
        dynamic_beta[is_perfect] = 0.98
        
        # 使用动态 beta 进行更新
        self.domain_weights.copy_(dynamic_beta * self.domain_weights + (1 - dynamic_beta) * new_weights)
        # ============================================================

        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'update_domain_weights'):
            self.trainer.datamodule.update_domain_weights(self.domain_weights.cpu().clone())

        if self.trainer.is_global_zero and self.current_epoch % 10 == 0:
            print(f"\n📊 [TSD Epoch {self.current_epoch}] 域难度评估报告 (退火温度: {dynamic_temp:.3f}, 进度: {progress*100:.1f}%)")
            print(f" {'域':^4} | {'难度':^8} | {'权重':^8} | {'分类损失':^8} | {'样本数':^6}")
            for d in range(self.hparams.num_domains):
                if self._local_sample_count[d] > 0:
                    print(f" {d:02d}   | {domain_difficulty[d]:8.4f} | {self.domain_weights[d]:8.3f} "
                          f"| {avg_cls_loss[d]:8.4f} | {int(self._local_sample_count[d]):6d}")
            print()

        self._local_cls_loss_sum.zero_()
        self._local_kd_gap_sum.zero_()
        self._local_sample_count.zero_()

    def validation_step(self, batch, batch_idx):
        loss, _, preds, targets, _, _, _ = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, _, preds, targets, cls_loss_vec, per_sample_kd_gap, domain_ids = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        # 【核心修改】：无视 ddk_enabled 开关，只要有 domain_ids，全收集！
        output = {
            "preds": preds,
            "targets": targets,
        }
        if domain_ids is not None:
            output["domain_ids"] = domain_ids
            # Baseline 可能没有逐样本的 cls_loss_vec，给它填 0 防止维度对齐报错
            output["cls_loss"] = cls_loss_vec if cls_loss_vec is not None else torch.zeros_like(preds, dtype=torch.float)
                
        self.test_step_outputs.append(output)

    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return

        # 1. 局部拼接
        local_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        local_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
        has_domains = "domain_ids" in self.test_step_outputs[0]
        
        if has_domains:
            local_domains = torch.cat([x["domain_ids"] for x in self.test_step_outputs])
            local_cls_loss = torch.cat([x["cls_loss"] for x in self.test_step_outputs])

        # 2. 全局多卡收集 (all_gather)
        all_preds = self.all_gather(local_preds).view(-1)
        all_targets = self.all_gather(local_targets).view(-1)
        if has_domains:
            all_domains = self.all_gather(local_domains).view(-1)
            all_cls_loss = self.all_gather(local_cls_loss).view(-1)

        # 3. 主进程打印与保存
        if self.trainer.is_global_zero:
            all_preds = all_preds.cpu()
            all_targets = all_targets.cpu()
            
            import json
            class_metrics = {}
            num_classes = self.hparams.net.data_attributes.class_num
            
            for c in range(num_classes):
                mask = (all_targets == c)
                if mask.sum() > 0:
                    acc = (all_preds[mask] == all_targets[mask]).sum().item() / mask.sum().item()
                    class_metrics[int(c)] = {"acc": float(acc)}
                else:
                    class_metrics[int(c)] = {"acc": 0.0}
            
            save_path = "baseline_class_metrics.json"
            with open(save_path, "w") as f:
                json.dump(class_metrics, f, indent=4)
            print(f"\n✅ [DDK 预处理] 错题本已成功生成并保存至: {save_path}")

            # =======================================================
            # 【核心修改】：无视 ddk_enabled 限制，只要有域信息就强制打印报告！
            # =======================================================
            if has_domains:
                all_domains = all_domains.cpu()
                all_cls_loss = all_cls_loss.cpu()
                
                print(f"\n📊 [Baseline vs DDK 对比评估] 细粒度域评估报告")
                print(f" {'域':^4} | {'准确率':^8} | {'分类损失':^8} | {'样本数':^6}")
                
                domain_stats = []
                # 注意：你之前设置的是 num_domains=10，这里会完整打印有数据的域
                for d in range(self.hparams.num_domains):
                    mask = (all_domains == d)
                    total = mask.sum().item()
                    if total > 0:
                        acc = (all_preds[mask] == all_targets[mask]).sum().item() / total
                        avg_cls = all_cls_loss[mask].mean().item()
                        print(f" {d:02d}   | {acc*100:6.2f}% | {avg_cls:8.4f} | {total:6d}")
                        domain_stats.append(acc)

                if domain_stats:
                    print(f"\n 📈 汇总 -> 平均准确率: {np.mean(domain_stats)*100:.2f}%, 准确率标准差: {np.std(domain_stats)*100:.2f}%\n")
            
        self.test_step_outputs.clear()

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    pass