import os
import urllib.request
import tarfile

def download_oxford102_for_ddk(root_dir="./data/kd_datasets/10_Oxford102Flowers"):
    """
    下载并解压 Oxford 102 Flowers 数据集，保持官方原始结构，
    以完美适配自定义的 Dataset 类和 DDK 域划分逻辑。
    """
    os.makedirs(root_dir, exist_ok=True)
    
    # 官方下载链接
    files = {
        "102flowers.tgz": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz",
        "imagelabels.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat",
        "setid.mat": "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat"
    }
    
    # 下载文件
    for filename, url in files.items():
        filepath = os.path.join(root_dir, filename)
        if not os.path.exists(filepath):
            print(f"正在下载 {filename} (可能需要几分钟)...")
            urllib.request.urlretrieve(url, filepath)
            print(f"{filename} 下载完成！")
        else:
            print(f"{filename} 已存在，跳过下载。")
            
    # 解压图片压缩包
    # 官方的 tgz 文件解压后会自动生成一个包含所有图片的 'jpg' 文件夹
    tgz_path = os.path.join(root_dir, "102flowers.tgz")
    jpg_dir = os.path.join(root_dir, "jpg")
    
    if not os.path.exists(jpg_dir):
        print("\n正在解压图片文件 102flowers.tgz...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=root_dir)
        print("解压完成！")
    else:
        print("\n图片文件夹 'jpg' 已存在，跳过解压。")
        
    print(f"\n✅ 数据集目录结构准备完毕！")
    print(f"请将你的域标签文件放在此处: {os.path.join(root_dir, 'ddk_image_domain_labels.txt')}")

if __name__ == "__main__":
    download_oxford102_for_ddk()