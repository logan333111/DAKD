import json
import numpy as np
import os

# 1. 读取刚刚跑出来的错题本
with open('baseline_class_metrics.json', 'r') as f:
    class_metrics = json.load(f)

# 2. 按照准确率升序排序 (准确率越低的狗品种，难度越大，越排在前面)
sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1]['acc'])

# 3. 均匀切分为 10 个难度域 (Domain 0 为最难品种，Domain 9 为最易品种)
num_domains = 10
chunks = np.array_split(sorted_classes, num_domains)

class_to_domain = {}
print(f"{'='*50}\n📊 [Dog 类别分域报告] 共 {len(class_metrics)} 种狗，划分为 {num_domains} 个域\n{'='*50}")

for domain_id, chunk in enumerate(chunks):
    acc_list = [item[1]['acc'] for item in chunk]
    print(f"Domain {domain_id:02d} | 包含 {len(chunk):3d} 个品种 | 平均准确率: {np.mean(acc_list)*100:.2f}%")
    for item in chunk:
        class_id = str(item[0]) # 类别 ID
        class_to_domain[class_id] = int(domain_id)

# 4. 保存为 json 文件
save_path = 'class_to_domain_10.json'
with open(save_path, 'w') as f:
    json.dump(class_to_domain, f, indent=4)
    
print(f"\n✅ 映射表已生成: {save_path}")
print("👉 下一步：请把这个 json 文件移动到你 Stanford Dogs 数据集的根目录下！")