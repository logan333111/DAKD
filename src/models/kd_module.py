from typing import Any, Dict, Tuple
import torch
import torch.nn.functional as F  # [DDK] 需要用到 softmax
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
        # [DDK 新增] 接收 DDK 配置参数
        ddk_enabled: bool = True,
        num_domains: int = 10,
        ddk_alpha: float = 0.5,
    ) -> None:
        super().__init__()

        # 保存超参
        self.save_hyperparameters(logger=False)

        self.net = net
        self.use_teacher = use_teacher
        if use_teacher:
            self.kd_criterion = kd_criterion
        
        # 基础分类损失
        # 加入 label_smoothing=0.1，极大地缓解模型在训练后期的盲目自信和过拟合
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)

        # [DDK 新增] 初始化 DDK 核心组件
        self.ddk_enabled = ddk_enabled
        if self.ddk_enabled:
            # 1. 域权重 r
            self.register_buffer('domain_weights', torch.ones(num_domains))
            
            # 2. 【终极防卡死方案】：用 EMA(指数移动平均) 状态替代 Accumulator
            self.register_buffer('domain_loss_ema', torch.zeros(num_domains))

        # --- 以下是原有的指标定义 (保持不变) ---
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
        
        # [计划 B 新增] 用于暂存 test 阶段的预测数据，用于生成领域报告
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_best.reset()

    # [DDK 修改] model_step 接收的数据变了，多了一个 domain_ids
    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ):
        def calculate_loss_weights(current_epoch, total_epochs):
            weight1 = current_epoch / (total_epochs * 8)
            weight1 = np.clip(weight1, 0, 1)
            weight2 = 1 - weight1
            # 【防过拟合策略】：限制蒸馏 Loss 的最低权重为 0.2
            weight2 = np.clip(weight2, 0.2, 1)
            return weight1, weight2

        x, y, domain_ids = batch
        loss_dict = {}
        
        if self.use_teacher:
            outputs = self.forward(x)
            img_loss, kd_loss = self.kd_criterion(outputs) 
            cls_loss_vec = self.criterion(outputs[1], y)
            cls_loss_weight, kd_loss_weight = calculate_loss_weights(self.current_epoch, self.trainer.max_epochs)
            
            if self.ddk_enabled:
                sample_weights = self.domain_weights[domain_ids]
                weighted_cls_loss = (cls_loss_vec * sample_weights).mean()
                
                batch_avg_weight = sample_weights.mean()
                weighted_img_loss = img_loss * batch_avg_weight
                weighted_kd_loss = kd_loss * batch_avg_weight
                
                # 【终极防卡死：无通信实时更新】
                if self.training:
                    with torch.no_grad():
                        for d in range(self.hparams.num_domains):
                            mask = (domain_ids == d)
                            if mask.sum() > 0:
                                curr_loss = cls_loss_vec[mask].mean()
                                
                                if self.domain_loss_ema[d] == 0:
                                    self.domain_loss_ema[d] = curr_loss
                                else:
                                    self.domain_loss_ema[d] = 0.9 * self.domain_loss_ema[d] + 0.1 * curr_loss
                        
                        # 【恢复稳健的 DDK 退火策略】
                        progress = self.current_epoch / self.trainer.max_epochs
                        dynamic_temperature = 2.0 + 10.0 * progress 
                        
                        new_weights = F.softmax(self.domain_loss_ema / dynamic_temperature, dim=0) * self.hparams.num_domains
                        self.domain_weights.copy_(0.9 * self.domain_weights + 0.1 * new_weights)
            else:
                weighted_cls_loss = cls_loss_vec.mean()
                weighted_img_loss = img_loss
                weighted_kd_loss = kd_loss

            final_loss = cls_loss_weight * weighted_cls_loss + \
                         (kd_loss_weight * weighted_img_loss + kd_loss_weight * weighted_kd_loss) / 2
            
            preds = torch.argmax(outputs[1], dim=1)
            loss_dict["loss"] = final_loss
            loss_dict["cls_loss"] = cls_loss_vec.mean()
            loss_dict["img_loss"] = img_loss
            loss_dict["kd_loss"] = kd_loss

        else:
            # 没有 Teacher 的情况
            logits = self.forward(x)
            loss_vec = self.criterion(logits, y)
            
            if self.ddk_enabled:
                sample_weights = self.domain_weights[domain_ids]
                final_loss = (loss_vec * sample_weights).mean()
                
                if self.training:
                    with torch.no_grad():
                        for d in range(self.hparams.num_domains):
                            mask = (domain_ids == d)
                            if mask.sum() > 0:
                                curr_loss = loss_vec[mask].mean()
                                if self.domain_loss_ema[d] == 0:
                                    self.domain_loss_ema[d] = curr_loss
                                else:
                                    self.domain_loss_ema[d] = 0.9 * self.domain_loss_ema[d] + 0.1 * curr_loss
                        
                        # 【恢复稳健的 DDK 退火策略】
                        progress = self.current_epoch / self.trainer.max_epochs
                        dynamic_temperature = 2.0 + 10.0 * progress 
                        
                        new_weights = F.softmax(self.domain_loss_ema / dynamic_temperature, dim=0) * self.hparams.num_domains
                        self.domain_weights.copy_(0.9 * self.domain_weights + 0.1 * new_weights)
            else:
                final_loss = loss_vec.mean()
                
            preds = torch.argmax(logits, dim=1)
            loss_dict["loss"] = final_loss

        return final_loss, loss_dict, preds, y

    def training_step(self, batch, batch_idx):
        loss, loss_dicts, preds, targets = self.model_step(batch)

        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.use_teacher:
            self.cls_loss(loss_dicts["cls_loss"])
            self.img_loss(loss_dicts["img_loss"])
            self.kd_loss(loss_dicts["kd_loss"])
            self.log("train/cls_loss", self.cls_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/img_loss", self.img_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/kd_loss", self.kd_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss

    # [DDK 核心] 仅保留日志打印，彻底消灭通信死锁
    def on_train_epoch_end(self) -> None:
        if self.ddk_enabled and self.trainer.is_global_zero:
            # 只在 GPU 0 上打印，看一眼当前实时权重即可
            top_weights = self.domain_weights.topk(3).values.cpu().numpy().round(3)
            print(f"\n[DDK Epoch {self.current_epoch}] Real-time Top Domain Weights: {top_weights}")

    def validation_step(self, batch, batch_idx):
        loss, loss_dicts, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

    # [计划 B 新增] 获取预测结果并存入列表
    def test_step(self, batch, batch_idx):
        loss, loss_dicts, preds, targets = self.model_step(batch)
        
        # 解包拿到 domain_ids
        x, y, domain_ids = batch
        
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        
        # 将结果转到 CPU 并暂存
        self.test_step_outputs.append({
            "preds": preds.detach().cpu(),
            "targets": targets.detach().cpu(),
            "domain_ids": domain_ids.detach().cpu()
        })

    # [计划 B 新增] 在测试结束时汇总并打印每个 Domain 的准确率
    def on_test_epoch_end(self) -> None:
        if not self.test_step_outputs:
            return
            
        # 如果是多卡测试，只在主卡上打印报告，防止重复刷屏
        if self.trainer.is_global_zero:
            # 由于可能有多张卡，我们这里简单汇总当前卡收集到的数据。
            # （为了绝对精确的全局统计，通常需 all_gather，但对于看趋势这已经足够精准了）
            all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
            all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])
            all_domain_ids = torch.cat([x["domain_ids"] for x in self.test_step_outputs])

            print(f"\n{'='*55}")
            print(f"📊 [Test Report] 细粒度域准确率分析 (Per-Domain Accuracy)")
            print(f"{'='*55}")

            for d in range(self.hparams.num_domains):
                mask = (all_domain_ids == d)
                total_samples = mask.sum().item()
                
                if total_samples > 0:
                    domain_preds = all_preds[mask]
                    domain_targets = all_targets[mask]
                    
                    correct = (domain_preds == domain_targets).sum().item()
                    acc = correct / total_samples
                    
                    print(f"Domain {d:02d} | 准确率: {acc*100:6.2f}% | 测试样本数: {total_samples}")

            print(f"{'='*55}\n")
            
        # 清理内存
        self.test_step_outputs.clear()

    # ... (setup 和 configure_optimizers 保持不变) ...
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
    # 测试代码
    pass