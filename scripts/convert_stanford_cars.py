#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将已下载的 HuggingFace Stanford Cars (Arrow 格式) 转换为项目所需的格式：
  - data/kd_datasets/6_StanfordCars/cars/images/{split}_{index:05d}.jpg
  - data/kd_datasets/6_StanfordCars/cars/train.csv  (列: label,filename)
  - data/kd_datasets/6_StanfordCars/cars/test.csv
  - data/kd_datasets/6_StanfordCars/cars/classes.txt  (类别名列表)

用法:
    python scripts/convert_stanford_cars.py
"""
import os
import argparse
from datasets import load_from_disk
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="转换 Stanford Cars 数据为项目格式")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="data/kd_datasets",
        help="HuggingFace Arrow 数据所在目录 (包含 train/ 和 test/ 子目录)",
    )
    parser.add_argument(
        "--dst_dir",
        type=str,
        default="data/kd_datasets/6_StanfordCars/cars",
        help="目标输出目录",
    )
    return parser.parse_args()


def convert_split(dataset, split, dst_dir, class_names=None):
    """转换一个 split 的数据"""
    img_dir = os.path.join(dst_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    csv_rows = []  # (label, filename)

    for idx, example in enumerate(tqdm(dataset, desc=f"转换 {split}")):
        image = example["image"]
        label = example["label"]

        # 保存图片
        filename = f"{split}_{idx:05d}.jpg"
        img_path = os.path.join(img_dir, filename)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(img_path, quality=95)

        csv_rows.append((label, filename))

    # 写 CSV
    csv_path = os.path.join(dst_dir, f"{split}.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("label,filename\n")
        for label, filename in csv_rows:
            f.write(f"{label},{filename}\n")

    print(f"  ✅ {split}: {len(csv_rows)} 张图片, CSV 保存到 {csv_path}")

    # 返回类别名（仅从 train 中提取一次）
    if class_names is None and hasattr(dataset.features["label"], "names"):
        return dataset.features["label"].names
    return class_names


def main():
    args = parse_args()
    os.makedirs(args.dst_dir, exist_ok=True)

    # 加载已下载的 Arrow 数据
    train_src = os.path.join(args.src_dir, "train")
    test_src = os.path.join(args.src_dir, "test")

    print(f"正在加载 Arrow 数据...")
    print(f"  训练集: {train_src}")
    print(f"  测试集: {test_src}")

    train_dataset = load_from_disk(train_src)
    test_dataset = load_from_disk(test_src)

    print(f"  训练集大小: {len(train_dataset)}")
    print(f"  测试集大小: {len(test_dataset)}")

    # 转换
    class_names = convert_split(train_dataset, "train", args.dst_dir)
    convert_split(test_dataset, "test", args.dst_dir, class_names)

    # 保存类别名列表
    if class_names:
        classes_path = os.path.join(args.dst_dir, "classes.txt")
        with open(classes_path, "w", encoding="utf-8") as f:
            for i, name in enumerate(class_names):
                f.write(f"{i} {name}\n")
        print(f"  ✅ 类别名保存到 {classes_path} ({len(class_names)} 类)")

    print(f"\n转换完成！输出目录: {args.dst_dir}")


if __name__ == "__main__":
    main()
