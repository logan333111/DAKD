import os
import shutil

# 1. 注入你最新的 Kaggle Token
os.environ['KAGGLE_API_TOKEN'] = 'KGAT_b5f97fc95c032ef6dafae0b71ec93f11'

print("🚀 开始使用最新版 KaggleHub 下载 NABirds...")
try:
    import kagglehub
except ImportError:
    print("❌ 请先运行: pip install kagglehub")
    exit()

# 2. 下载并解压 (kagglehub 自带极速下载和精美进度条)
# 注意：它会先安全地下载并解压到系统的默认缓存目录
print("⏬ 正在下载并解压，请耐心等待进度条跑完...")
cache_path = kagglehub.dataset_download("duyminhle/nabirds")
print(f"\n✅ 下载解压成功！Kaggle缓存路径为: {cache_path}")

# 3. 自动将数据无缝迁移到你的 VL2Lite 目录
target_dir = "/data2/hgq/VL2Lite/VL2Lite-main/data/kd_datasets/2_NABirds"
os.makedirs(target_dir, exist_ok=True)

print(f"📦 正在将近五万张图片迁移到你的目标目录 (大概需要十几秒)...")
for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
        
print(f"🎉 大功告成！NABirds 数据集已完美部署在: {target_dir}")