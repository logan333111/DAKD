#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
============================================================
基于 CLIP 文本特征的 Stanford Cars 零样本域划分
============================================================
策略 (模仿 generate_dogs_domains_text.py):
  1. 从类别名中提取品牌 (e.g. "BMW", "Audi", "Chevrolet")
  2. 用 CLIP 文本编码器获取每个品牌名称的语义嵌入
  3. 对品牌文本特征做 K-means 聚类 → N 个域
  4. 为所有图片分配域标签，写入 ddk_image_domain_labels.txt

用法：
    python scripts/generate_cars_domains_text.py --num_domains 10 --device cuda:0
"""
import os
import argparse
import numpy as np
import torch
import csv
from collections import defaultdict
from sklearn.cluster import KMeans

import open_clip


def extract_brand(class_name):
    """
    从类名中提取品牌名称。
    例如: "am general hummer suv 2000" -> "AM General"
          "bmw 1 series convertible 2012" -> "BMW"
          "mercedes-benz c-class sedan 2012" -> "Mercedes-Benz"
    """
    name = class_name.strip()
    # 已知的多词品牌 (小写匹配)
    multi_word_brands = [
        "am general", "aston martin", "land rover", "rolls-royce",
        "mercedes-benz",
    ]
    name_lower = name.lower()
    for brand in multi_word_brands:
        if name_lower.startswith(brand):
            return brand.title()

    # 默认取第一个单词作为品牌
    return name.split()[0].upper() if name else "Unknown"


def extract_text_features(brands, model, tokenizer, device):
    """提取品牌名称的 CLIP 文本特征"""
    print(f"\n[特征提取] 正在提取 {len(brands)} 个汽车品牌的语义特征...")

    prompts = [f"A photo of a {brand} car, an automobile." for brand in brands]
    text_tokens = tokenizer(prompts).to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)

    print(f"  ✅ 文本特征提取完成，形状: {text_features.shape}")
    return text_features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='基于 CLIP 文本特征的 Stanford Cars 自适应域划分')
    parser.add_argument('--root_dir', type=str,
                        default='data/kd_datasets/6_StanfordCars/cars',
                        help='Stanford Cars 数据集根目录')
    parser.add_argument('--teacher_arch', type=str, default='convnext_xxlarge',
                        help='使用的 CLIP 模型架构')
    parser.add_argument('--teacher_pretrained', type=str,
                        default='src/models/teacher/open_clip_pytorch_model.bin',
                        help='CLIP 模型权重路径')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='目标域数量')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的 GPU')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🔬 基于 CLIP 文本特征的 Stanford Cars 零样本域划分")
    print(f"{'='*70}")
    print(f"  数据集: {args.root_dir}")
    print(f"  模型: {args.teacher_arch}")
    print(f"  目标域数: {args.num_domains}")
    print(f"{'='*70}\n")

    # 1. 读取类别名
    classes_file = os.path.join(args.root_dir, 'classes.txt')
    if not os.path.exists(classes_file):
        raise FileNotFoundError(f"找不到文件: {classes_file}。请先运行 convert_stanford_cars.py 。")

    class_id_to_name = {}
    with open(classes_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            class_id_to_name[int(parts[0])] = parts[1]

    print(f"  ✅ 成功读取 {len(class_id_to_name)} 个类别名")

    # 2. 提取品牌列表
    class_to_brand = {}
    brands_set = set()
    for cid, cname in class_id_to_name.items():
        brand = extract_brand(cname)
        class_to_brand[cid] = brand
        brands_set.add(brand)

    brands = sorted(list(brands_set))
    brand_to_idx = {b: i for i, b in enumerate(brands)}
    print(f"  ✅ 成功提取 {len(brands)} 个品牌: {', '.join(brands[:10])}...")

    # 3. 加载 CLIP 模型
    print("\n[加载模型] 加载 CLIP 模型以获取 Text Encoder...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained=args.teacher_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.teacher_arch)
    model = model.to(args.device)

    # 4. 提取文本特征
    text_features = extract_text_features(brands, model, tokenizer, args.device)

    del model
    torch.cuda.empty_cache()

    # 5. K-Means 聚类
    print(f"\n[域划分] 正在 K-Means 聚类 {len(brands)} 个品牌特征 → {args.num_domains} 个域...")
    kmeans = KMeans(n_clusters=args.num_domains, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(text_features)

    brand_to_domain = {}
    for i, brand in enumerate(brands):
        brand_to_domain[brand] = int(cluster_labels[i])

    # 6. 输出域语义分析
    domain_to_brands = defaultdict(list)
    for brand, did in brand_to_domain.items():
        domain_to_brands[did].append(brand)

    print(f"\n{'='*70}")
    print(f"📊 域语义聚类结果 (依据 CLIP 理解的汽车品牌特性)")
    print(f"{'='*70}")

    semantics_file = os.path.join(args.root_dir, 'ddk_domain_semantics.csv')
    with open(semantics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain ID', 'Brand Count', 'Brands'])
        for did in range(args.num_domains):
            brand_list = sorted(domain_to_brands[did])
            examples = ", ".join(brand_list)
            print(f"  域 {did:02d} ({len(brand_list):3d} 个品牌) : {examples}")
            writer.writerow([did, len(brand_list), examples])

    # 7. 为所有图片分配 Domain ID
    print("\n[生成标签] 正在给所有图片分配域标签...")
    label_file = os.path.join(args.root_dir, 'ddk_image_domain_labels.txt')

    total_count = 0
    all_entries = []

    for split in ['train', 'test']:
        csv_path = os.path.join(args.root_dir, f'{split}.csv')
        if not os.path.exists(csv_path):
            continue
        with open(csv_path, 'r', encoding='utf-8') as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split(',', 1)
                label = int(parts[0])
                filename = parts[1]
                brand = class_to_brand.get(label, "Unknown")
                domain_id = brand_to_domain.get(brand, 0)
                all_entries.append((filename, domain_id))
                total_count += 1

    with open(label_file, 'w', encoding='utf-8') as f:
        for filename, domain_id in all_entries:
            f.write(f"{filename} {domain_id}\n")

    print(f"  ✅ 域标签已保存到 {label_file}")
    print(f"  ✅ 共 {len(all_entries)} 条记录")

    # 8. 打印域统计
    print(f"\n{'='*70}")
    print(f"🎯 最终域统计")
    print(f"{'='*70}")
    domain_counts = defaultdict(int)
    for _, did in all_entries:
        domain_counts[did] += 1

    for did in sorted(domain_counts.keys()):
        bar = '█' * (domain_counts[did] // 50)
        print(f"  域 {did:02d}: {domain_counts[did]:5d} 张  {bar}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
