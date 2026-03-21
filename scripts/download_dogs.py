import os
import urllib.request
import tarfile

def download_and_extract(url, download_root, extract_root, filename):
    file_path = os.path.join(download_root, filename)
    if not os.path.exists(file_path):
        print(f"📥 正在下载 {filename} (可能需要一些时间)...")
        urllib.request.urlretrieve(url, file_path)
        print(f"✅ 下载完成: {filename}")
    else:
        print(f"⏩ 文件已存在，跳过下载: {filename}")
        
    print(f"📦 正在解压 {filename} 到 {extract_root}...")
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=extract_root)
    print(f"✅ 解压完成: {filename}")

def setup_stanford_dogs():
    data_root = './data/kd_datasets/5_StanfordDogs'
    os.makedirs(data_root, exist_ok=True)
    
    print(f"🚀 开始连接斯坦福官方节点下载数据...")
    print(f"📂 目标路径: {data_root}")
    
    # URLs for Stanford Dogs Dataset
    images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    lists_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar"
    
    # 1. 下载并解压 Images
    download_and_extract(
        url=images_url,
        download_root=data_root,
        extract_root=data_root,
        filename="images.tar"
    )
    
    # 2. 下载并解压 train_list.mat 和 test_list.mat
    download_and_extract(
        url=lists_url,
        download_root=data_root,
        extract_root=data_root,
        filename="lists.tar"
    )
    
    print("\n✅ 数据集下载与解压全部完成！")
    print(f"� 请检查 {data_root}/Images 目录是否包含图片。")
    print(f"� 请检查 {data_root}/train_list.mat 和 test_list.mat 是否存在。")

if __name__ == "__main__":
    setup_stanford_dogs()