#!/bin/bash

# === Baseline Training (multi-dataset sweep) ===
# python3 train.py -m data/attributes=0_CUB_200_2011,1_FGVC_AIRCRAFT,2_NABirds model.use_teacher=true,false

# === Evaluation Example ===
CUDA_VISIBLE_DEVICES=0,1 python src/eval.py \
    trainer=ddp \
    model.net.student.arch=resnet18 \
    model.use_teacher=true \
    model.ddk_enabled=false \
    logger=tensorboard \
    trainer.devices=2 \
    data/attributes=1_FGVC_AIRCRAFT \
    ckpt_path=logs/train/runs/YOUR_RUN/checkpoints/epoch_099.ckpt

# === DAKD Training Example ===
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py \
    model.net.student.arch=resnet18 \
    data/attributes=8_CALTECH256 \
    model.ddk_enabled=true \
    trainer=ddp \
    model.use_teacher=true \
    logger=tensorboard

# === Ablation: TSD cls_only ===
CUDA_VISIBLE_DEVICES=0,3,4,6 python src/train.py \
    trainer=ddp \
    model.use_teacher=true \
    logger=tensorboard \
    data/attributes=0_CUB_200_2011 \
    model.tsd_alpha=1.0 \
    trainer.devices=4 \
    "+callbacks.ablation_logger._target_=src.callbacks.ablation_logger.AblationLogger" \
    "+callbacks.ablation_logger.experiment_name=exp1_tsd_metric" \
    "+callbacks.ablation_logger.variant_name=cls_only"

# === Ablation: TSD kd_gap_only ===
CUDA_VISIBLE_DEVICES=0,3,4,6 python src/train.py \
    trainer=ddp \
    model.use_teacher=true \
    logger=tensorboard \
    data/attributes=0_CUB_200_2011 \
    model.tsd_alpha=0.0 \
    trainer.devices=4 \
    "+callbacks.ablation_logger._target_=src.callbacks.ablation_logger.AblationLogger" \
    "+callbacks.ablation_logger.experiment_name=exp1_tsd_metric" \
    "+callbacks.ablation_logger.variant_name=kd_gap_only"
