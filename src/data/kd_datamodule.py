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

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = train_dataset.dataloader
        self.data_val: Optional[Dataset] = test_dataset.dataloader
        self.data_test: Optional[Dataset] = test_dataset.dataloader

        self.batch_size_per_device = batch_size

        # Domain sampling weights, updated by KDModule at each epoch end. None = uniform sampling.
        self._domain_weights: Optional[torch.Tensor] = None

    @property
    def num_classes(self) -> int:
        return self.data_train.attributes.num_classes

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def update_domain_weights(self, weights: torch.Tensor) -> None:
        """Update domain sampling weights. Called by KDModule.on_train_epoch_end()."""
        self._domain_weights = weights

    def train_dataloader(self) -> DataLoader[Any]:
        """Create training DataLoader with optional weighted sampling."""
        sampler = None
        shuffle = True

        # NOTE: Weighted sampling is disabled to avoid double-reweighting with loss reweighting.
        # Kept here for reference if sampling-based reweighting is needed.
        # if self._domain_weights is not None and hasattr(self.data_train, 'domains'):
        #     domains = torch.tensor(self.data_train.domains, dtype=torch.long)
        #     sample_weights = self._domain_weights[domains].double()
        #     sampler = WeightedRandomSampler(
        #         weights=sample_weights,
        #         num_samples=len(self.data_train),
        #         replacement=True,
        #     )
        #     shuffle = False

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
        """Save DataModule state to checkpoint."""
        state = {}
        if self._domain_weights is not None:
            state['domain_weights'] = self._domain_weights
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Restore DataModule state from checkpoint."""
        if 'domain_weights' in state_dict:
            self._domain_weights = state_dict['domain_weights']
            print("[DAR] Restored domain weights from checkpoint")


if __name__ == "__main__":
    _ = KDDataModule()
