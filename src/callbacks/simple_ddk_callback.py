# ====================================================================================
# 简化的DDK回调 - simple_ddk_callback.py
# 
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  核心作用：在每个验证epoch结束后，协调模型和数据模块更新DDK采样权重             ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
#
# 【在DDK系统中的位置】
# SimpleDDKCallback 是连接 SimpleDDKModule 和 SimpleDDKDataModule 的桥梁：
# 
#   Model验证完成 → Callback获取损失 → DataModule更新采样权重 → 下轮训练更新采样
#
# 【阅读建议】
# 这是DDK策略的"大脑"，负责何时和如何更新采样权重
# ====================================================================================

from typing import Any, Dict, List, Optional
from lightning import Callback, LightningModule, Trainer


class SimpleDDKCallback(Callback):
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                  SimpleDDKCallback - DDK采样权重更新回调                      ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    【核心功能】
    1. 在验证epoch结束后获取分域损失
    2. 调用数据模块更新DDK采样权重
    3. 管理checkpoint中的DDK状态
    
    【工作流程】
    ┌────────────────────────────────────────────────────────────────────┐
    │  on_validation_epoch_end()                                          │
    │    │                                                                 │
    │    ├── 1. 检查是否超过warmup期                                      │
    │    │                                                                 │
    │    ├── 2. 从Model获取学生/教师损失                                  │
    │    │      model.get_ddk_losses() → ([s_loss], [t_loss])            │
    │    │                                                                 │
    │    ├── 3. 调用DataModule更新权重                                    │
    │    │      datamodule.update_ddk_weights(s_losses, t_losses)         │
    │    │                                                                 │
    │    └── 4. 记录DDK日志                                               │
    └────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        update_interval: int = 1,  # 每N个epoch更新一次
        warmup_epochs: int = 0,    # 预热epoch数（期间不更新权重）
    ):
        """
        参数说明:
        update_interval: 更新间隔（epoch数）
                         1表示每个epoch都更新
                         2表示每隔一个epoch更新
        warmup_epochs: 预热期
                       期间使用均匀采样，不启用DDK动态权重
                       目的是让模型先稳定训练一段时间再开始DDK
        """
        super().__init__()
        self.update_interval = update_interval
        self.warmup_epochs = warmup_epochs
        
        print(f"[SimpleDDKCallback] 初始化完成:")
        print(f"  - 更新间隔: 每 {update_interval} 个epoch")
        print(f"  - 预热期: {warmup_epochs} 个epoch")
    
    def on_validation_epoch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule
    ) -> None:
        """
        验证epoch结束时更新DDK权重
        
        【触发时机】
        每个验证epoch结束后自动调用
        
        【执行条件】
        1. 已过预热期 (current_epoch >= warmup_epochs)
        2. 满足更新间隔条件
        """
        current_epoch = trainer.current_epoch
        
        # 预热期不更新
        if current_epoch < self.warmup_epochs:
            print(f"[SimpleDDKCallback] Epoch {current_epoch}: 预热期，跳过DDK更新")
            return
        
        # 检查更新间隔
        if (current_epoch - self.warmup_epochs) % self.update_interval != 0:
            return
        
        # 获取数据模块
        datamodule = trainer.datamodule
        if not hasattr(datamodule, 'update_ddk_weights'):
            print(f"[SimpleDDKCallback] 警告: 数据模块不支持DDK更新")
            return
        
        # 获取DDK损失
        if hasattr(pl_module, 'get_ddk_losses'):
            student_losses, teacher_losses = pl_module.get_ddk_losses()
        else:
            # 尝试从日志中获取
            student_losses = self._get_losses_from_logs(trainer, 'ddk/student_val_loss_domain')
            teacher_losses = self._get_losses_from_logs(trainer, 'ddk/teacher_val_loss_domain')
        
        if not student_losses or not teacher_losses:
            print(f"[SimpleDDKCallback] Epoch {current_epoch}: 无法获取损失值，跳过DDK更新")
            return
        
        # 更新DDK权重
        print(f"\n[SimpleDDKCallback] Epoch {current_epoch}: 更新DDK差异因子")
        print(f"  - 学生验证损失: {[f'{l:.4f}' for l in student_losses]}")
        print(f"  - 教师验证损失: {[f'{l:.4f}' for l in teacher_losses]}")
        
        datamodule.update_ddk_weights(student_losses, teacher_losses)
        
        # 记录到日志
        sampling_info = datamodule.get_current_sampling_info()
        for i, (weight, factor) in enumerate(zip(
            sampling_info['domain_weights'], 
            sampling_info['discrepancy_factors']
        )):
            pl_module.log(f'ddk/domain_{i}_weight', weight)
            pl_module.log(f'ddk/domain_{i}_discrepancy', factor)
    
    def _get_losses_from_logs(
        self, 
        trainer: Trainer, 
        prefix: str
    ) -> List[float]:
        """
        备用方法：从训练日志中获取损失值
        
        当Model没有get_ddk_losses方法时使用
        """
        losses = []
        logged_metrics = trainer.logged_metrics
        
        i = 0
        while f'{prefix}_{i}' in logged_metrics:
            losses.append(float(logged_metrics[f'{prefix}_{i}']))
            i += 1
        
        return losses
    
    def on_save_checkpoint(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        checkpoint: Dict[str, Any]
    ) -> None:
        """
        保存DDK状态到checkpoint
        
        【作用】
        确保训练中断后恢复时，DDK的采样权重能够继续使用，
        而不是从头开始计算
        """
        datamodule = trainer.datamodule
        if hasattr(datamodule, 'state_dict'):
            checkpoint['ddk_datamodule_state'] = datamodule.state_dict()
    
    def on_load_checkpoint(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        checkpoint: Dict[str, Any]
    ) -> None:
        """
        从checkpoint加载DDK状态
        
        【作用】
        恢复训练时，加载之前保存的DDK采样权重，
        使训练能够继续使用之前学习到的采样策略
        """
        if 'ddk_datamodule_state' in checkpoint:
            datamodule = trainer.datamodule
            if hasattr(datamodule, 'load_state_dict'):
                datamodule.load_state_dict(checkpoint['ddk_datamodule_state'])
                print("[SimpleDDKCallback] 已从checkpoint恢复DDK状态")
