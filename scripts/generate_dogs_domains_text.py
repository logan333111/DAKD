import os
import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.cluster import KMeans
import scipy.io
import csv

import open_clip

def extract_text_features(breeds, model, tokenizer, device):
    """提取 120 个狗品种名称的 CLIP 文本特征"""
    print(f"\n[特征提取] 正在提取 {len(breeds)} 种狗的语义特征...")
    
    # 构造 Prompt
    prompts = [f"A photo of a {breed}, a type of dog." for breed in breeds]
    
    # Tokenize (可能需要截断)
    text_tokens = tokenizer(prompts).to(device)
    
    model.eval()
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # L2 归一化
        text_features = text_features / (text_features.norm(dim=-1, keepdim=True) + 1e-10)
        
    print(f"  ✅ 文本特征提取完成，形状: {text_features.shape}")
    return text_features.cpu().numpy()

def main():
    parser = argparse.ArgumentParser(description='基于 CLIP 文本特征的 StanfordDogs 自适应域划分')
    parser.add_argument('--root_dir', type=str, 
                        default='data/kd_datasets/5_StanfordDogs',
                        help='StanfordDogs 数据集根目录')
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
    print(f"🔬 基于 CLIP 文本特征的 StanfordDogs 零样本域划分")
    print(f"{'='*70}")
    print(f"  数据集: {args.root_dir}")
    print(f"  模型: {args.teacher_arch}")
    print(f"  目标域数: {args.num_domains}")
    print(f"{'='*70}\n")
    
    # 1. 解析品种名称 (共有 120 个)
    img_dir = os.path.join(args.root_dir, 'Images')
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"找不到的目录: {img_dir}。请先下载并解压数据。")
        
    folder_names = sorted(os.listdir(img_dir))
    if len(folder_names) != 120:
        print(f"[警告] 期望 120 个类别文件夹，但找到了 {len(folder_names)} 个")
        
    # 解析出干净的名字：n02085620-Chihuahua -> Chihuahua (并把下划线换空格)
    breeds = []
    folder_to_breed = {}
    for folder in folder_names:
        clean_name = folder.split('-')[-1].replace('_', ' ').strip()
        breeds.append(clean_name)
        folder_to_breed[folder] = clean_name
        
    print(f"  ✅ 成功解析 {len(breeds)} 种狗的名称")
    print(f"     例如: {breeds[0]}, {breeds[1]}...")
    
    # 2. 加载 CLIP 模型和 Tokenizer
    print("\n[加载模型] 加载 CLIP 模型以获取 Text Encoder...")
    model, _, _ = open_clip.create_model_and_transforms(
        args.teacher_arch, pretrained=args.teacher_pretrained
    )
    tokenizer = open_clip.get_tokenizer(args.teacher_arch)
    model = model.to(args.device)
    
    # 3. 提取特征
    text_features = extract_text_features(breeds, model, tokenizer, args.device)
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    # 4. K-Means 聚类
    print(f"\n[域划分] 正在 K-Means 聚类 120 个语义特征 → {args.num_domains} 个域...")
    kmeans = KMeans(n_clusters=args.num_domains, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(text_features)
    
    # 构建 文件夹名 -> Domain ID 映射
    folder_to_domain = {}
    for i, folder in enumerate(folder_names):
        folder_to_domain[folder] = cluster_labels[i]
        
    # 5. 输出域语义分析
    domain_to_breeds = defaultdict(list)
    for folder, did in folder_to_domain.items():
        domain_to_breeds[did].append(folder_to_breed[folder])
        
    print(f"\n{'='*70}")
    print(f"📊 域语义聚类结果 (依据 CLIP 理解的狗品种特性)")
    print(f"{'='*70}")
    
    semantics_file = os.path.join(args.root_dir, 'ddk_domain_semantics.csv')
    with open(semantics_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Domain ID', 'Breed Count', 'Example Breeds'])
        for did in range(args.num_domains):
            breed_list = sorted(domain_to_breeds[did])
            examples = ", ".join(breed_list[:5]) + ("..." if len(breed_list) > 5 else "")
            print(f"  域 {did:02d} ({len(breed_list):3d} 个类) : {examples}")
            writer.writerow([did, len(breed_list), ", ".join(breed_list)])
            
    # 6. 为所有的图片分配 Domain ID 并输出到 label 文件
    # StanfordDogs 的 dataloader 使用的是 filename (例如 "n02085620-Chihuahua/n02085620_10074.jpg")
    print("\n[生成标签] 正在给所有图片分配标签...")
    label_file = os.path.join(args.root_dir, 'ddk_image_domain_labels.txt')
    
    # 读取所有用到的图片
    all_files = []
    for split in ['train', 'test']:
        mat_path = os.path.join(args.root_dir, f"{split}_list.mat")
        if os.path.exists(mat_path):
            mat_data = scipy.io.loadmat(mat_path)
            file_list = [item[0][0] for item in mat_data['file_list']]
            all_files.extend(file_list)
            
    # 去重
    all_files = sorted(list(set(all_files)))
    print(f"  发现图库总图片数: {len(all_files)}")
    
    with open(label_file, 'w', encoding='utf-8') as f:
        for filename in all_files:
            folder = os.path.dirname(filename)
            did = folder_to_domain.get(folder, 0)
            f.write(f"{filename} {did}\n")
            
    print(f"  ✅ 域标签已保存到 {label_file}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
