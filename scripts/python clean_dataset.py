import os
from pathlib import Path

# 你的数据集根目录路径（请根据实际情况确认是否正确）
ROOT_DIR = "/data2/hgq/VL2Lite/VL2Lite-main/data/kd_datasets/8_CALTECH256/256_ObjectCategories"

# 定义我们要保留的合法图片后缀名（全小写）
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# 安全开关：如果 DRY_RUN 为 True，则只打印准备删除的文件名，不会真的删除。
# 确认没问题后，把 DRY_RUN 改为 False 再运行一次。
DRY_RUN = True 

def clean_dataset(root_path):
    root_path = Path(root_path)
    
    if not root_path.exists():
        print(f"错误：找不到路径 {root_path}")
        return

    deleted_count = 0
    
    print(f"开始扫描目录: {root_path}")
    print("-" * 50)
    
    # 遍历所有子目录和文件
    for subdir, _, files in os.walk(root_path):
        for file in files:
            file_path = Path(subdir) / file
            
            # 获取文件后缀名并转为小写
            extension = file_path.suffix.lower()
            
            # 如果后缀名不在我们定义的合法列表中
            if extension not in VALID_EXTENSIONS:
                if DRY_RUN:
                    print(f"[试运行 - 准备删除] {file_path}")
                    deleted_count += 1
                else:
                    try:
                        file_path.unlink() # 执行删除操作
                        print(f"[已删除] {file_path}")
                        deleted_count += 1
                    except Exception as e:
                        print(f"删除失败 {file_path}: {e}")

    print("-" * 50)
    if DRY_RUN:
        print(f"试运行结束。共发现 {deleted_count} 个非图片文件准备删除。")
        print("如果上面的列表看起来没问题，请将代码中的 DRY_RUN = False，然后再运行一次以执行真正的删除。")
    else:
        print(f"清理完成！共删除了 {deleted_count} 个非图片文件。")

if __name__ == "__main__":
    clean_dataset(ROOT_DIR)