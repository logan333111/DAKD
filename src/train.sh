#!/bin/bash
python3 train.py -m data/attributes=2_NABirds,3_DTD,4_OxfordIIITPet,5_StanfordDogs,6_StanfordCars,7_CALTECH101,8_CALTECH256,9_GTSRB model.use_teacher=true,false

#0_CUB_200_2011,1_FGVC_AIRCRAFT,
CUDA_VISIBLE_DEVICES=0,5,6,7 python src/eval.py  trainer=ddp model.use_teacher=true logger=tensorboard ckpt_path=logs/train/runs/2026-03-17_20-28-44/checkpoints/epoch_066.ckpt
CUDA_VISIBLE_DEVICES=0,5,6,7 python src/train.py  trainer=ddp model.use_teacher=true logger=tensorboard 