#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载 Stanford Cars 数据集 (从 Hugging Face Hub)
用法:
    python scripts/download_stanford_cars.py --save_dir /data2/hgq/datasets/stanford_cars
"""

import argparse
import os

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="下载 Stanford Cars 数据集")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/data2/hgq/datasets/stanford_cars",
        help="数据集保存路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    print("正在从 Hugging Face 加载 Stanford Cars 训练集...")
    train_dataset = load_dataset(
        "Multimodal-Fatima/StanfordCars_train", split="train"
    )

    print("正在从 Hugging Face 加载 Stanford Cars 测试集...")
    test_dataset = load_dataset(
        "Multimodal-Fatima/StanfordCars_test", split="test"
    )

    print(f"\n训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    print(f"特征: {train_dataset.features}")

    # 查看第一个样本
    example = train_dataset[0]
    print(f"第一个样本标签: {example['label']}")
    print(f"图片大小: {example['image'].size}")

    # 保存到本地磁盘
    train_save_path = os.path.join(args.save_dir, "train")
    test_save_path = os.path.join(args.save_dir, "test")

    print(f"\n正在保存训练集到 {train_save_path} ...")
    train_dataset.save_to_disk(train_save_path)

    print(f"正在保存测试集到 {test_save_path} ...")
    test_dataset.save_to_disk(test_save_path)

    print("\n数据集下载并保存完成！")
    print(f"保存路径: {args.save_dir}")
    print("后续加载方式:")
    print(f"  from datasets import load_from_disk")
    print(f'  train = load_from_disk("{train_save_path}")')
    print(f'  test  = load_from_disk("{test_save_path}")')


if __name__ == "__main__":
    main()
