"""Delayed ModelCheckpoint: skip saving during early epochs to avoid noisy checkpoints."""
from lightning.pytorch.callbacks import ModelCheckpoint


class DelayedModelCheckpoint(ModelCheckpoint):
    """Only save best checkpoints after `start_epoch` epochs."""

    def __init__(self, start_epoch: int = 25, **kwargs):
        super().__init__(**kwargs)
        self.start_epoch = start_epoch

    def _should_skip_saving_checkpoint(self, trainer) -> bool:
        if trainer.current_epoch < self.start_epoch:
            return True
        return super()._should_skip_saving_checkpoint(trainer)
