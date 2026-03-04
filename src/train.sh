#!/bin/bash
python3 train.py -m data/attributes=2_NABirds,3_DTD,4_OxfordIIITPet,5_StanfordDogs,6_StanfordCars,7_CALTECH101,8_CALTECH256,9_GTSRB model.use_teacher=true,false

#0_CUB_200_2011,1_FGVC_AIRCRAFT,

CUDA_VISIBLE_DEVICES=4 python src/eval.py logger=[] trainer=ddp model.use_teacher=true ckpt_path=logs/train/runs/2026-03-03_21-42-16/checkpoints/epoch_199.ckpt
CUDA_VISIBLE_DEVICES=4,5,6,7 python src/train.py  trainer=ddp model.use_teacher=true logger=tensorboard