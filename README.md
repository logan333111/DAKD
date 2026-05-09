______________________________________________________________________

<div align="center">

# DAKD: Difficulty-Aware Knowledge Distillation from Vision-Language Models

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://pytorchlightning.ai/">
  <img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white">
</a>
<a href="https://hydra.cc/">
  <img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd">
</a>
<a href="https://github.com/ashleve/lightning-hydra-template">
  <img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray">
</a>
<br>

[![Conference](http://img.shields.io/badge/NeurIPS%202026-Paper-4b44ce.svg)](#)

</div>

---

## Description

This repository contains the **official implementation** of the paper:

> **"Not All Data Are Equal: Difficulty-Aware Distillation from Vision-Language Models"**

DAKD is a **PyTorch Lightning + Hydra** framework for distilling knowledge from large frozen Vision-Language Models (e.g., CLIP) into lightweight student networks. 

---

## Features

- **Frozen VLM Teacher**: No teacher fine-tuning required  
- **Condensation Layers**: Reduce dimensionality for both image and text embeddings  
- **Multi-Loss**: Classification loss + Visual KD + Linguistic KD  
- **Difficulty-Aware Reweighting**: Dynamic domain-level optimisation weights  
- **Configurable**: Hydra-based setup for custom data, model, or experiment scripts

---

## Installation

1. **Clone project**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/DAKD.git
   cd DAKD
   ```

2. *(Optional)* **Create conda environment**:
   ```bash
   conda env create -f environment.yaml
   conda activate dakd
   ```

3. **Install PyTorch** per [official instructions](https://pytorch.org/get-started/) if not included in the environment.

---

## Data Setup

If your dataset is in a different path, create a soft link:
```bash
ln -s /your/dataset/path ./data/kd_datasets
```
Update `configs/data/` if needed.

---

## How to Run

We use [PyTorch Lightning](https://www.pytorchlightning.ai/) for training loops and [Hydra](https://hydra.cc/) for configuration.

### Basic Commands

- **Train on CPU**:
  ```bash
  python src/train.py trainer=cpu
  ```

- **Train on GPU**:
  ```bash
  python src/train.py trainer=gpu
  ```

- **Train with DAKD** (multi-GPU):
  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
      trainer=ddp trainer.devices=4 \
      model.use_teacher=true \
      model.ddk_enabled=true \
      model.net.student.arch=resnet18 \
      data/attributes=0_CUB_200_2011 \
      logger=tensorboard
  ```

- **Train Baseline** (without DAKD):
  ```bash
  python src/train.py \
      model.use_teacher=true \
      model.ddk_enabled=false \
      data/attributes=0_CUB_200_2011
  ```

### Experiment Configs

Pick a config from `configs/experiment/`:
```bash
python src/train.py experiment=experiment_name
```
and you can override any parameter:
```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

The default student backbone is `mobilenet_v2`. To override it from the command line:
```bash
python src/train.py model.net.student.arch=resnet18
```

*(See `src/train.sh` for example scripts.)*

---

## Configuration

- **trainer** configs in `configs/trainer/`  
- **data** configs in `configs/data/`  
- **model** configs in `configs/model/`  
- **experiment** configs in `configs/experiment/`  

Hydra allows combining or overriding these configs easily.

---

Built upon the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template).   
We thank open-source projects (PyTorch, Lightning, Hydra, OpenCLIP) that enable this work.

---

## Citation

If you use this code or find DAKD helpful, please cite:

```bibtex
@inproceedings{dakd2026,
  title={Not All Data Are Equal: Difficulty-Aware Distillation from Vision-Language Models},
  author={},
  booktitle={NeurIPS},
  year={2026}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
Please see the [LICENSE](LICENSE) file for details.
