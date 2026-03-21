"""
延迟保存最佳模型的 Checkpoint Callback。
前 N 轮不保存（last.ckpt 除外），之后每次发现更好的 val/acc 就保存并删除上一个。
"""
from lightning.pytorch.callbacks import ModelCheckpoint


class DelayedModelCheckpoint(ModelCheckpoint):
    """在指定 epoch 之后才开始保存最佳权重的 ModelCheckpoint。

    参数:
        start_epoch: 从第几轮开始保存（默认50），之前的轮次全部跳过
        其余参数同 ModelCheckpoint
    """

    def __init__(self, start_epoch: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        """前 start_epoch 轮跳过所有 checkpoint 保存"""
        if trainer.current_epoch < self.start_epoch:
            return True
        return super()._should_skip_saving_checkpoint(trainer)
