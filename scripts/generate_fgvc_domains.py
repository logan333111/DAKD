"""
============================================================
基于教师模型特征的自适应域划分脚本
============================================================
步骤：
1. 加载 CLIP 教师模型（ConvNeXt-XXL）
2. 遍历所有 FGVC-Aircraft 图片，提取教师视觉特征
3. 按 family（70 个）取平均特征
4. 对 family 级特征做 K-means 聚类 → 10 个域
5. 生成新的 ddk_image_domain_labels.txt

用法：
    python scripts/generate_fgvc_domains.py --num_domains 10 --device cuda:5
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from sklearn.cluster import KMeans

import open_clip


class FGVCSimpleDataset(Dataset):
    """仅用于特征提取的简易 FGVC-Aircraft 数据集"""
    def __init__(self, root_dir, split='trainval', transform=None):
        self.root_dir = root_dir
        self.image_paths = []
        self.image_ids = []
        self.families = []
        
        # 读取 family 标签
        family_file = os.path.join(root_dir, 'data', f'images_family_{split}.txt')
        with open(family_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                img_id = parts[0]
                family = parts[1] if len(parts) > 1 else 'unknown'
                self.image_ids.append(img_id)
                self.families.append(family)
                self.image_paths.append(
                    os.path.join(root_dir, 'data', 'images', img_id + '.jpg')
                )
        
        self.transform = transform
        # 构建 family → index 映射
        unique_families = sorted(set(self.families))
        self.family_to_idx = {f: i for i, f in enumerate(unique_families)}
        self.idx_to_family = {i: f for f, i in self.family_to_idx.items()}
        print(f"[域划分] 加载了 {len(self.image_paths)} 张图片，{len(unique_families)} 个 family")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        family_idx = self.family_to_idx[self.families[idx]]
        return image, family_idx, self.image_ids[idx]


def extract_features(model, dataloader, device):
    """提取所有图片的教师特征"""
    all_features = []
    all_family_ids = []
    all_image_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, family_ids, img_ids) in enumerate(dataloader):
            images = images.to(device)
            features = model.encode_image(images)
            # L2 归一化
            features = features / (features.norm(dim=-1, keepdim=True) + 1e-10)
            
            all_features.append(features.cpu().numpy())
            all_family_ids.extend(family_ids.numpy().tolist())
            all_image_ids.extend(img_ids)
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  已处理 {(batch_idx+1) * len(images)}/{len(dataloader.dataset)} 张图片")
    
    return np.concatenate(all_features), all_family_ids, all_image_ids


def cluster_families(features, family_ids, num_families, num_domains, idx_to_family):
    """按 family 取平均特征，然后 K-means 聚类"""
    # 1. 计算每个 family 的平均特征
    family_features = defaultdict(list)
    for feat, fid in zip(features, family_ids):
        family_features[fid].append(feat)
    
    family_avg = {}
    for fid, feats in family_features.items():
        family_avg[fid] = np.mean(feats, axis=0)
    
    # 2. 排序后组成矩阵
    sorted_fids = sorted(family_avg.keys())
    feature_matrix = np.stack([family_avg[fid] for fid in sorted_fids])
    
    print(f"\n[域划分] Family 特征矩阵形状: {feature_matrix.shape}")
    print(f"[域划分] 正在 K-means 聚类 {len(sorted_fids)} 个 family → {num_domains} 个域...")
    
    # 3. K-means 聚类
    kmeans = KMeans(n_clusters=num_domains, random_state=42, n_init=20, max_iter=500)
    cluster_labels = kmeans.fit_predict(feature_matrix)
    
    # 4. 构建 family_idx → domain_id 映射
    family_to_domain = {}
    for fid, cluster_id in zip(sorted_fids, cluster_labels):
        family_to_domain[fid] = cluster_id
    
    # 5. 打印聚类结果
    print(f"\n{'='*70}")
    print(f"📊 Family → Domain 聚类结果")
    print(f"{'='*70}")
    for domain_id in range(num_domains):
        families_in_domain = [
            idx_to_family[fid] for fid, did in family_to_domain.items() if did == domain_id
        ]
        count = sum(1 for fid in family_ids if family_to_domain.get(fid) == domain_id)
        print(f"  域 {domain_id:02d} ({count:4d} 张): {', '.join(sorted(families_in_domain))}")
    print(f"{'='*70}\n")
    
    return family_to_domain


def main():
    parser = argparse.ArgumentParser(description='基于教师特征的自适应域划分')
    parser.add_argument('--root_dir', type=str, 
                        default='data/kd_datasets/1_FGVC_AIRCRAFT',
                        help='FGVC-Aircraft 数据集根目录')
    parser.add_argument('--teacher_arch', type=str, default='convnext_xxlarge',
                        help='教师模型架构')
    parser.add_argument('--teacher_pretrained', type=str,
                        default='src/models/teacher/open_clip_pytorch_model.bin',
                        help='教师模型权重路径')
    parser.add_argument('--num_domains', type=int, default=10,
                        help='目标域数量')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='特征提取批大小')
    parser.add_argument('--device', type=str, default='cuda:5',
                        help='使用的 GPU')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🔬 基于教师特征的自适应域划分")
    print(f"{'='*70}")
    print(f"  数据集: {args.root_dir}")
    print(f"  教师模型: {args.teacher_arch}")
    print(f"  目标域数: {args.num_domains}")
    print(f"  设备: {args.device}")
    print(f"{'='*70}\n")
    
    # 1. 加载教师模型
    print("[步骤1] 加载教师模型...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained=args.teacher_pretrained
    )
    model = model.to(args.device)
    model.eval()
    print(f"  ✅ 模型加载完成")
    
    # 2. 准备数据集（用 train+val 的全部数据来做聚类）
    print("\n[步骤2] 加载数据集...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])
    
    # train + val 合并来做聚类特征提取
    dataset_train = FGVCSimpleDataset(args.root_dir, split='train', transform=transform)
    dataset_val = FGVCSimpleDataset(args.root_dir, split='val', transform=transform)
    
    # 合并两个数据集
    combined_dataset = torch.utils.data.ConcatDataset([dataset_train, dataset_val])
    dataloader = DataLoader(combined_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # 使用 train 数据集的映射表
    idx_to_family = dataset_train.idx_to_family
    num_families = len(idx_to_family)
    
    # 3. 提取特征
    print(f"\n[步骤3] 提取教师特征 ({len(combined_dataset)} 张图片)...")
    features, family_ids, image_ids = extract_features(model, dataloader, args.device)
    print(f"  ✅ 特征提取完成，形状: {features.shape}")
    
    # 释放 GPU 显存
    del model
    torch.cuda.empty_cache()
    
    # 4. K-means 聚类
    print(f"\n[步骤4] Family 级 K-means 聚类...")
    family_to_domain = cluster_families(
        features, family_ids, num_families, args.num_domains, idx_to_family
    )
    
    # 5. 生成域标签文件（覆盖所有 split：train + test）
    print("[步骤5] 生成域标签文件...")
    output_path = os.path.join(args.root_dir, 'data', 'ddk_image_domain_labels.txt')
    
    # 备份旧文件
    if os.path.exists(output_path):
        backup_path = output_path + '.bak_manufacturer'
        os.rename(output_path, backup_path)
        print(f"  ✅ 旧文件已备份到 {backup_path}")
    
    # 需要为 train + test 的所有图片生成标签
    total_count = 0
    with open(output_path, 'w') as out_f:
        for split in ['train', 'val', 'test']:
            family_file = os.path.join(args.root_dir, 'data', f'images_family_{split}.txt')
            if not os.path.exists(family_file):
                continue
            with open(family_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(' ', 1)
                    img_id = parts[0]
                    family = parts[1] if len(parts) > 1 else 'unknown'
                    # 获取 family 的 index，再查 domain
                    fid = dataset_train.family_to_idx.get(family, 0)
                    domain_id = family_to_domain.get(fid, 0)
                    out_f.write(f"{img_id} {domain_id}\n")
                    total_count += 1
    
    print(f"  ✅ 域标签已保存到 {output_path}")
    print(f"  ✅ 共 {total_count} 条记录")
    
    # 6. 打印域统计
    print(f"\n{'='*70}")
    print(f"🎯 最终域统计")
    print(f"{'='*70}")
    domain_counts = defaultdict(int)
    with open(output_path, 'r') as f:
        for line in f:
            _, did = line.strip().split()
            domain_counts[int(did)] += 1
    
    for did in sorted(domain_counts.keys()):
        bar = '█' * (domain_counts[did] // 20)
        print(f"  域 {did:02d}: {domain_counts[did]:5d} 张  {bar}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
