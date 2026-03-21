"""
============================================================
基于 CLIP 文本特征的 FGVC-Aircraft 零样本域划分
============================================================
策略 (模仿 generate_dogs_domains_text.py):
  1. 从数据集中解析出 70 个 family 名称
  2. 用 CLIP 文本编码器获取每个 family 名称的语义嵌入
  3. 对 70 个文本特征做 K-means 聚类 → N 个域
  4. 为所有图片分配域标签，写入 ddk_image_domain_labels.txt

优点：不需要加载图片，速度极快，完全零样本。

用法：
    python scripts/generate_fgvc_domains_text.py --num_domains 10 --device cuda:0
"""
import os
import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import KMeans
import csv

import open_clip


def extract_text_features(families, model, tokenizer, device):
    """提取所有 family 名称的 CLIP 文本特征"""
    print(f"\n[特征提取] 正在提取 {len(families)} 个飞机家族的语义特征...")

    # 构造 Prompt
    prompts = [f"A photo of a {family}, a type of aircraft." for family in families]

    # Tokenize
    text_tokens = tokenizer(prompts).to(device)

    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # L2 归一化
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)

    print(f"  ✅ 文本特征提取完成，形状: {text_features.shape}")
    return text_features.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='基于 CLIP 文本特征的 FGVC-Aircraft 自适应域划分')
    parser.add_argument('--root_dir', type=str,
                        default='data/kd_datasets/1_FGVC_AIRCRAFT',
                        help='FGVC-Aircraft 数据集根目录')
    parser.add_argument('--teacher_arch', type=str, default='convnext_xxlarge',
                        help='使用的 CLIP 模型架构 (借用文本编码器)')
    parser.add_argument('--teacher_pretrained', type=str,
                        default='src/models/teacher/open_clip_pytorch_model.bin',
                        help='CLIP 模型权重路径')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='目标域数量')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='使用的 GPU')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"🔬 基于 CLIP 文本特征的 FGVC-Aircraft 零样本域划分")
    print(f"{'='*70}")
    print(f"  数据集: {args.root_dir}")
    print(f"  模型: {args.teacher_arch}")
    print(f"  目标域数: {args.num_domains}")
    print(f"{'='*70}\n")

    # 1. 解析 family 名称 (共 70 个)
    # 从 images_family_trainval.txt 中提取所有唯一的 family 名
    data_dir = os.path.join(args.root_dir, 'data')
    family_file = os.path.join(data_dir, 'images_family_trainval.txt')
    if not os.path.exists(family_file):
        raise FileNotFoundError(f"找不到文件: {family_file}。请确认数据集路径。")

    families_set = set()
    with open(family_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) > 1:
                families_set.add(parts[1])

    families = sorted(list(families_set))
    print(f"  ✅ 成功解析 {len(families)} 个飞机家族名称")
    print(f"     例如: {families[0]}, {families[1]}, {families[2]}...")

    # 构建 family -> index 映射
    family_to_idx = {f: i for i, f in enumerate(families)}

    # 2. 加载 CLIP 模型和 Tokenizer
    print("\n[加载模型] 加载 CLIP 模型以获取 Text Encoder...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained=args.teacher_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.teacher_arch)
    model = model.to(args.device)

    # 3. 提取文本特征
    text_features = extract_text_features(families, model, tokenizer, args.device)

    # 清理显存
    del model
    torch.cuda.empty_cache()

    # 4. K-Means 聚类
    print(f"\n[域划分] 正在 K-Means 聚类 {len(families)} 个语义特征 → {args.num_domains} 个域...")
    kmeans = KMeans(n_clusters=args.num_domains, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(text_features)

    # 构建 family -> Domain ID 映射
    family_to_domain = {}
    for i, family in enumerate(families):
        family_to_domain[family] = int(cluster_labels[i])

    # 5. 输出域语义分析
    domain_to_families = defaultdict(list)
    for family, did in family_to_domain.items():
        domain_to_families[did].append(family)

    print(f"\n{'='*70}")
    print(f"📊 域语义聚类结果 (依据 CLIP 理解的飞机家族特性)")
    print(f"{'='*70}")

    semantics_file = os.path.join(data_dir, 'ddk_domain_semantics.csv')
    with open(semantics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain ID', 'Family Count', 'Families'])
        for did in range(args.num_domains):
            family_list = sorted(domain_to_families[did])
            examples = ", ".join(family_list[:5]) + ("..." if len(family_list) > 5 else "")
            print(f"  域 {did:02d} ({len(family_list):3d} 个家族) : {examples}")
            writer.writerow([did, len(family_list), ", ".join(family_list)])

    # 6. 为所有图片分配 Domain ID 并输出到 label 文件
    print("\n[生成标签] 正在给所有图片分配标签...")
    label_file = os.path.join(data_dir, 'ddk_image_domain_labels.txt')

    # 备份旧文件
    if os.path.exists(label_file):
        backup_path = label_file + '.bak_before_text'
        if not os.path.exists(backup_path):
            import shutil
            shutil.copy2(label_file, backup_path)
            print(f"  ✅ 旧文件已备份到 {backup_path}")

    # 遍历 train, val, test 的 family 文件，为每张图片写入域标签
    total_count = 0
    all_entries = []
    for split in ['train', 'val', 'test']:
        split_family_file = os.path.join(data_dir, f'images_family_{split}.txt')
        if not os.path.exists(split_family_file):
            continue
        with open(split_family_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                img_id = parts[0]
                family = parts[1] if len(parts) > 1 else 'unknown'
                domain_id = family_to_domain.get(family, 0)
                all_entries.append((img_id, domain_id))
                total_count += 1

    # 去重 (同一张图可能在 train 和 trainval 中出现)
    all_entries = sorted(list(set(all_entries)))

    with open(label_file, 'w', encoding='utf-8') as f:
        for img_id, domain_id in all_entries:
            f.write(f"{img_id} {domain_id}\n")

    print(f"  ✅ 域标签已保存到 {label_file}")
    print(f"  ✅ 共 {len(all_entries)} 条记录 (去重后)")

    # 7. 打印域统计
    print(f"\n{'='*70}")
    print(f"🎯 最终域统计")
    print(f"{'='*70}")
    domain_counts = defaultdict(int)
    for _, did in all_entries:
        domain_counts[did] += 1

    for did in sorted(domain_counts.keys()):
        bar = '█' * (domain_counts[did] // 20)
        print(f"  域 {did:02d}: {domain_counts[did]:5d} 张  {bar}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
