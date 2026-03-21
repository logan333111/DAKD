from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
    random_split,
)
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

class KDDataModule(LightningDataModule):

    def __init__(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        attributes,
        data_name,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = train_dataset.dataloader
        self.data_val: Optional[Dataset] = test_dataset.dataloader
        self.data_test: Optional[Dataset] = test_dataset.dataloader

        self.batch_size_per_device = batch_size

        # ============================================================
        # [DAR - Sampling Reweighting] 域采样权重
        # 由 KDModule 在每个 epoch 结束后通过 update_domain_weights() 更新
        # None 表示使用均匀随机采样（默认行为）
        # ============================================================
        self._domain_weights: Optional[torch.Tensor] = None

    @property
    def num_classes(self) -> int:
        return self.data_train.attributes.num_classes

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def update_domain_weights(self, weights: torch.Tensor) -> None:
        """
        [DAR] 更新域采样权重。
        由 KDModule.on_train_epoch_end() 在每个 epoch 结束后调用。

        参数:
            weights: 域权重张量，形状为 (num_domains,)，
                     较大的值表示该域更难、需要更多采样
        """
        self._domain_weights = weights

    def train_dataloader(self) -> DataLoader[Any]:
        """
        创建训练 DataLoader。

        [DAR - Sampling Reweighting 路径]
        当域权重可用且数据集支持 domain_id 时，使用 WeightedRandomSampler
        让难域的样本在每个 epoch 中被采样更多次。
        否则退回标准的均匀随机 shuffle。
        """
        sampler = None
        shuffle = True

        # [DAR] 如果有域权重且数据集提供了域信息，启用加权采样
        # ⚠️ 注意: 为防止“双重惩罚”(Double Reweighting)，我们在 Loss 层保留了域权重，
        # 在采样层暂时关闭加权随机采样。保证模型能均匀地看到所有的简单样本，
        # 同时利用 Loss Reweighting 在困难样本上拉大梯度。
        # if self._domain_weights is not None and hasattr(self.data_train, 'domains'):
        #     domains = torch.tensor(self.data_train.domains, dtype=torch.long)
        #     # 每个样本的采样权重 = 其所属域的权重
        #     sample_weights = self._domain_weights[domains].double()
        #     sampler = WeightedRandomSampler(
        #         weights=sample_weights,
        #         num_samples=len(self.data_train),  # 保持每个 epoch 的总样本数不变
        #         replacement=True,  # 允许重复采样（难域样本可能被采多次）
        #     )
        #     shuffle = False  # WeightedRandomSampler 与 shuffle 互斥

        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """保存 DataModule 状态到 checkpoint。"""
        state = {}
        if self._domain_weights is not None:
            state['domain_weights'] = self._domain_weights
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """从 checkpoint 恢复 DataModule 状态。"""
        if 'domain_weights' in state_dict:
            self._domain_weights = state_dict['domain_weights']
            print("[DAR] 已从 checkpoint 恢复域采样权重")


if __name__ == "__main__":
    _ = KDDataModule()
