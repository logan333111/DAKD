from typing import Any
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import random
import scipy.io
import PIL


class RandomCropAndResize(object):
    def __init__(self, min_crop_size, max_crop_size, final_size):
        self.min_crop_size = min_crop_size
        self.max_crop_size = max_crop_size
        self.final_size = final_size

    def __call__(self, img):
        # Set random crop size
        crop_size = random.randint(self.min_crop_size, self.max_crop_size)

        # RandomCrop 
        img = transforms.RandomCrop(crop_size)(img)

        # Final resize
        img = transforms.Resize((self.final_size, self.final_size))(img)

        return img


class KDDataset(Dataset):
    def __init__(self, data_name, data_root, attributes, split='train'):
        self.root_dir = data_root
        self.split = split
        self.transform = self.get_transform(split)
        self.dataloader = self.get_dataloader(data_name)
        self.attributes = attributes #prompt_tmpl, class_num, classes

    def get_transform(self, split):
        # 注意：这里原作者使用的是 CLIP/VLM 特有的 Normalize 参数，非常关键，我们保持绝对不变！
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])       
            
        if split == 'train':
            return transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                
                # 【核心替换】：用一句话的 RandAugment，替代原代码里那一大堆手动写的增强！
                # num_ops=2 表示每次随机选 2 种操作，magnitude=9 表示中等偏上的增强强度
                transforms.RandAugment(num_ops=2, magnitude=6),
                
                transforms.ToTensor(),
                normalize,
                # 原作者注释掉的 RandomErasing (随机擦除) 其实防过拟合也很好，
                # 如果加上 RandAugment 还是过拟合，你可以把下面这行取消注释
                # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                ])
        else:
            return transforms.Compose([
                        # 验证集保持干净，不做任何干扰
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ])

    def get_dataloader(self, data_name):
        if data_name == '0_CUB_200_2011':
            return CUB200Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '1_FGVC_AIRCRAFT':
            return AircraftDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '2_NABirds':
            return NABirdsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '3_DTD':
            return DTDDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '4_OxfordIIITPet':
            return OxfordIIPetDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '5_StanfordDogs':
            return StanfordDogsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '6_StanfordCars':
            return StanfordCarsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '7_CALTECH101':
            return Caltech101Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '8_CALTECH256':
            return Caltech256Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '9_GTSRB':
            return GTSRBTestDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '10_Oxford102Flowers':
            return Oxford102FlowersDataset(self.root_dir, split=self.split, transform=self.transform)

    def __len__(self):
        return len(self.dataloader)
        
    def __getitem__(self, idx):
        return self.dataloader[idx]
    



class GTSRBTestDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        if split == 'train':
            csv_file = os.path.join(root_dir, 'Train.csv')
        else:
            csv_file = os.path.join(root_dir, 'Test.csv')

        self.data_frame = pd.read_csv(csv_file, skiprows=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.root_dir + '/' + self.data_frame.iloc[idx, 7].strip()
        image = Image.open(img_path)
        label = int(self.data_frame.iloc[idx, 6])

        if self.transform:
            image = self.transform(image)

        return image, label


import os
import random
from PIL import Image
from torch.utils.data import Dataset

class CUB200Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split

        # 加载 train/test split
        splits = []
        with open(os.path.join(root_dir, 'train_test_split.txt'), 'r') as f:
            for line in f:
                _, is_train = line.strip().split(' ')
                splits.append(int(is_train))
        
        # 加载图片路径
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for idx, line in enumerate(f):
                img_id, img_path = line.strip().split(' ')
                # 根据 split 筛选
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.image_paths.append(os.path.join(root_dir, 'images', img_path))
                
        # 加载图片类别标签
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for idx, line in enumerate(f):
                _, label = line.strip().split(' ')
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.labels.append(int(label))
        
        # =======================================================
        # [DDK 新增] 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            # 仅在训练集打印一次，避免刷屏
            if split == "train":
                print(f"✅ [DDK] 成功加载 CUB200 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有鸟类品种默认分配到 Domain 0")
            self.class_domain_map = {}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # 注意：有些环境可能需要 .convert('RGB') 来处理灰度图
        image = Image.open(img_path).convert('RGB')
        
        # CUB原始标签是从1开始的，这里转换为从0开始 (0~199)
        label = self.labels[idx] - 1 
        
        # =======================================================
        # [DDK 新增] 直接根据真实的鸟类标签 (0~199) 获取 domain_id
        # =======================================================
        # 注意：假设 json 文件里的 key 是从 "0" 到 "199" 的字符串
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id


class DTDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): DTD root directory
            split (str): "train" 또는 "test".
            transform (callable, optional): transform.
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        img_path = os.path.join(root_dir, 'images')
        
        splits = {}

        split_file = os.path.join(root_dir, 'train_test_split.txt')
        if not os.path.isfile(split_file):
            with open(split_file, 'w') as f:
                # [安全修复] 使用 sorted() 保证每次生成的类别顺序一致！
                for category in sorted(os.listdir(img_path)):
                    category_path = os.path.join(img_path, category)
                    if os.path.isdir(category_path):
                        for img_name in sorted(os.listdir(category_path)):
                            if img_name.endswith('.jpg'):
                                if random.random() < 0.8:
                                    f.write(f"{img_name} train\n")
                                else:
                                    f.write(f"{img_name} test\n")

        with open(split_file, 'r') as f:
            for line in f:
                img_name, s = line.strip().split()
                splits[img_name] = s

        # [安全修复] 同样加上 sorted()，严格对齐 YAML 里的 0~46 类别顺序
        for label, category in enumerate(sorted(os.listdir(img_path))):
            category_path = os.path.join(img_path, category)
            if os.path.isdir(category_path):
                for img_name in sorted(os.listdir(category_path)):
                    if img_name.endswith('.jpg') and splits[img_name] == split:
                        self.image_paths.append(os.path.join(category_path, img_name))
                        self.labels.append(label)

        # =======================================================
        # [DDK 新增] 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            # 仅在训练集打印一次，避免刷屏
            if split == "train":
                print(f"✅ [DDK] 成功加载 DTD 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有材质品种默认分配到 Domain 0")
            self.class_domain_map = {}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 新增] 直接根据真实的材质标签 (0~46) 获取 domain_id
        # =======================================================
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id


import os
import json
from PIL import Image
from torch.utils.data import Dataset

class AircraftDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ["train", "test", "val"], "split should be 'train', 'test', or 'val'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        # 1. 标签映射 (获取类别名称到 class_id 的映射)
        label_map = {}
        with open(os.path.join(root_dir, 'data', 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                label_map[class_name] = idx

        # =======================================================
        # [DDK 新增] 2. 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            # 仅在训练集打印一次，避免刷屏
            if split == "train":
                print(f"✅ [DDK] 成功加载 Aircraft 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有飞机品种默认分配到 Domain 0")
            self.class_domain_map = {}

        # 3. 读取图片切分文件并加载路径与标签
        with open(os.path.join(root_dir, 'data', f'images_variant_{split}.txt'), 'r') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if len(line_split) == 2:
                    img_name, class_name = line.strip().split()
                else:
                    img_name = line_split[0]
                    class_name = ' '.join(line_split[1:])

                self.image_paths.append(os.path.join(root_dir, 'data', 'images', img_name + '.jpg'))
                self.labels.append(label_map[class_name])
                
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 新增] 直接根据真实的飞机器种标签获取 domain_id
        # =======================================================
        # 注意：json 字典的 key 是字符串，所以这里用 str(label) 查询
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id

class NABirdsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # 1. 读取图片路径
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_paths.append(os.path.join(root_dir, 'images', img_path))

        # 2. 读取标签
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                _, label = line.strip().split()
                # 原始标签减 1，转换为 0-based 索引
                self.labels.append(int(label) - 1)  
                
        # =======================================================
        # [DDK 新增] 3. 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            # 仅在训练集打印一次，避免刷屏
            if split == "train":
                print(f"✅ [DDK] 成功加载 NABirds 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有鸟类品种默认分配到 Domain 0")
            self.class_domain_map = {}
        
        # 4. 根据 train/test 划分过滤数据
        split_list = []
        with open(os.path.join(root_dir, 'train_test_split.txt'), 'r') as f:
            for line in f:
                _, is_train = line.strip().split()
                split_list.append(is_train)

        if split == "train":
            self.image_paths = [self.image_paths[i] for i, s in enumerate(split_list) if s == '1']
            self.labels = [self.labels[i] for i, s in enumerate(split_list) if s == '1']
        else:
            self.image_paths = [self.image_paths[i] for i, s in enumerate(split_list) if s == '0']
            self.labels = [self.labels[i] for i, s in enumerate(split_list) if s == '0']

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 新增] 直接根据真实的鸟类品种标签获取 domain_id
        # =======================================================
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id

class OxfordIIPetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Oxford-IIIT Pet 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}

        if split == "train":
            split_file = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            split_file = os.path.join(root_dir, "annotations", "test.txt")

        # split 파일에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                img_name, label_id, _, _ = line.strip().split()
                img_path = os.path.join(root_dir, "images", img_name + ".jpg")
                if img_name.split('_')[0] not in self.label_to_idx:
                    self.label_to_idx[img_name.split('_')[0]] = len(self.label_to_idx)
                self.image_paths.append(img_path)
                self.labels.append(self.label_to_idx[img_name.split('_')[0]])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CalTech256Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): CalTech256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # 파일이 없으면 새로 생성
        if not os.path.exists(split_file):
            with open(split_file, 'w') as file:
                categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith("BACKGROUND")])
                for category in categories:
                    category_path = os.path.join(root_dir, category)
                    images_in_category = sorted([img for img in os.listdir(category_path) if img.endswith('.jpg')])
                    random.shuffle(images_in_category)
                    
                    split_idx = int(len(images_in_category) * 0.8)
                    train_images = images_in_category[:split_idx]
                    test_images = images_in_category[split_idx:]
                    
                    for img in train_images:
                        file.write(f"train {os.path.join(category, img)} {category}\n")
                    for img in test_images:
                        file.write(f"test {os.path.join(category, img)} {category}\n")
        
        # train_test_split.txt에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                mode, img_path, category = line.strip().split()
                if mode == split:
                    if category not in self.label_to_idx:
                        self.label_to_idx[category] = len(self.label_to_idx)
                    self.image_paths.append(os.path.join(root_dir, img_path))
                    self.labels.append(self.label_to_idx[category])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CalTech101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): CalTech101 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # 파일이 없으면 새로 생성
        if not os.path.exists(split_file):
            with open(split_file, 'w') as file:
                categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
                for category in categories:
                    category_path = os.path.join(root_dir, category)
                    images_in_category = sorted([img for img in os.listdir(category_path) if img.endswith('.jpg')])
                    random.shuffle(images_in_category)
                    
                    split_idx = int(len(images_in_category) * 0.8)
                    train_images = images_in_category[:split_idx]
                    test_images = images_in_category[split_idx:]
                    
                    for img in train_images:
                        file.write(f"train {os.path.join(category, img)} {category}\n")
                    for img in test_images:
                        file.write(f"test {os.path.join(category, img)} {category}\n")
        
        # train_test_split.txt에서 이미지 경로와 레이블 정보 읽기
        with open(split_file, 'r') as file:
            for line in file:
                mode, img_path, category = line.strip().split()
                if mode == split:
                    if category not in self.label_to_idx:
                        self.label_to_idx[category] = len(self.label_to_idx)
                    self.image_paths.append(os.path.join(root_dir, img_path))
                    self.labels.append(self.label_to_idx[category])
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class StanfordCarsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Stanford Cars 数据集根目录.
            split (str): "train" 或 "test".
            transform (callable, optional): 图像变换.
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.df = pd.read_csv(os.path.join(root_dir, f"{split}.csv"))
        self.img_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.domains = []  # [DDK 新增] 用于存储每张图的 Domain ID
        
        # [DDK 新增] 预先加载 Domain 映射表 (filename -> domain_id)
        domain_map = {}
        domain_file_path = os.path.join(root_dir, 'ddk_image_domain_labels.txt')
        if os.path.exists(domain_file_path):
            with open(domain_file_path, 'r') as f:
                for line in f:
                    filename, dom_id = line.strip().split()
                    domain_map[filename] = int(dom_id)
        else:
            print(f"[Warning] DDK domain file not found at {domain_file_path}. Defaulting domains to 0.")
        
        # 为每张图片匹配 Domain ID
        for idx in range(len(self.df)):
            filename = self.df.iloc[idx, 1]  # filename 列
            self.domains.append(domain_map.get(filename, 0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 1])  # filename 列
        image = Image.open(img_name).convert('RGB')
        label = int(self.df.iloc[idx, 0])  # label 列
        
        # [DDK 新增] 获取 Domain ID
        domain_id = self.domains[idx]
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回值增加 domain_id
        return image, label, domain_id


class Oxford102FlowersDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Oxford 102 Flowers 数据集根目录 (需包含 jpg 文件夹及 .mat 文件)
            split (str): "train", "val" 或 "test"
            transform (callable, optional): 图像变换
        """
        assert split in ["train", "val", "test"], "split should be 'train', 'val', or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        # 1. 加载官方的 .mat 标签文件
        # imagelabels.mat 包含 1~102 的标签，需减 1 转换为 0-based (0~101)
        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))
        all_labels = labels_mat['labels'][0] - 1  
        
        # 2. 加载数据集划分文件
        # setid.mat 包含 trnid (train), valid (val), tstid (test)
        setid_mat = scipy.io.loadmat(os.path.join(root_dir, 'setid.mat'))
        if split == 'train':
            # 细粒度任务中通常将 train 和 val 合并以增加训练数据量
            split_ids = setid_mat['trnid'][0].tolist() + setid_mat['valid'][0].tolist()
        elif split == 'val':
            split_ids = setid_mat['valid'][0].tolist()
        else: # test
            split_ids = setid_mat['tstid'][0].tolist()
            
        # =======================================================
        # [DDK 新增] 3. 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"✅ [DDK] 成功加载 Flowers 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有花卉品种默认分配到 Domain 0")
            self.class_domain_map = {}
            
        # 4. 构建图像路径和标签列表
        img_dir = os.path.join(root_dir, 'jpg')
        for img_id in split_ids:
            # 官方图片命名格式为 image_00001.jpg (img_id 是 1-based)
            img_name = f"image_{img_id:05d}.jpg"
            self.image_paths.append(os.path.join(img_dir, img_name))
            
            # all_labels 的索引是 0-based，所以用 img_id - 1
            self.labels.append(all_labels[img_id - 1]) 
            
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 修改] 直接根据真实的花卉品种标签 (0~101) 获取 domain_id
        # =======================================================
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
            
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id       
class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Stanford Dogs 数据集根目录
            split (str): "train" 或 "test".
            transform (callable, optional): 图像增强.
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.split_mat = scipy.io.loadmat(os.path.join(root_dir, f"{split}_list.mat"))
        self.file_list = [item[0][0] for item in self.split_mat['file_list']]
        self.labels = [item[0] - 1 for item in self.split_mat['labels']]  # Convert 1-based index to 0-based
        
        self.img_dir = os.path.join(root_dir, "Images")
        self.transform = transform
        
        # =======================================================
        # [DDK 新增] 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            # 仅在训练集打印一次，避免刷屏
            if split == "train":
                print(f"✅ [DDK] 成功加载 Dog 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有狗品种默认分配到 Domain 0")
            self.class_domain_map = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 修改] 直接根据真实的狗品种标签 (0~119) 获取 domain_id
        # =======================================================
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id


class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Caltech 256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.imgs_path = os.path.join(root_dir, '101_ObjectCategories')
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, d))])
        self.transform = transform
        self.split = split
        
        train_indices = []
        test_indices = []
        
        # Load images and labels by class and split within each class
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.imgs_path, class_name)
            class_image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            class_labels = [class_idx] * len(class_image_paths)
            
            class_train_size = int(0.8 * len(class_image_paths))
            class_train_indices = list(range(len(self.image_paths), len(self.image_paths) + class_train_size))
            class_test_indices = list(range(len(self.image_paths) + class_train_size, len(self.image_paths) + len(class_image_paths)))
            
            self.image_paths.extend(class_image_paths)
            self.labels.extend(class_labels)
            
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        if not os.path.exists(split_file):
            with open(split_file, 'w') as f:
                for idx in train_indices:
                    f.write(f"train,{self.image_paths[idx]}\n")
                for idx in test_indices:
                    f.write(f"test,{self.image_paths[idx]}\n")
        
        split_indices = train_indices if split == "train" else test_indices
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class Caltech256Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Caltech 256 데이터셋의 디렉터리 경로.
            split (str): "train" 또는 "test".
            transform (callable, optional): 적용할 변환(Optional).
        """
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.imgs_path = os.path.join(root_dir, '256_ObjectCategories')
        self.image_paths = []
        self.labels = []
        self.class_names = sorted([d for d in os.listdir(self.imgs_path) if os.path.isdir(os.path.join(self.imgs_path, d))])
        self.transform = transform
        self.split = split
        
        train_indices = []
        test_indices = []
        
        # Load images and labels by class and split within each class
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.imgs_path, class_name)
            class_image_paths = [os.path.join(class_dir, img_name) for img_name in os.listdir(class_dir)]
            class_labels = [class_idx] * len(class_image_paths)
            
            class_train_size = int(0.8 * len(class_image_paths))
            class_train_indices = list(range(len(self.image_paths), len(self.image_paths) + class_train_size))
            class_test_indices = list(range(len(self.image_paths) + class_train_size, len(self.image_paths) + len(class_image_paths)))
            
            self.image_paths.extend(class_image_paths)
            self.labels.extend(class_labels)
            
            train_indices.extend(class_train_indices)
            test_indices.extend(class_test_indices)
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        if not os.path.exists(split_file):
            with open(split_file, 'w') as f:
                for idx in train_indices:
                    f.write(f"train,{self.image_paths[idx]}\n")
                for idx in test_indices:
                    f.write(f"test,{self.image_paths[idx]}\n")
        
        split_indices = train_indices if split == "train" else test_indices
        self.image_paths = [self.image_paths[i] for i in split_indices]
        self.labels = [self.labels[i] for i in split_indices]

        # =======================================================
        # [DDK 新增] 预先加载基于类别的 Domain 映射表 (class_id -> domain_id)
        # =======================================================
        domain_json_path = os.path.join(root_dir, 'class_to_domain_10.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"✅ [DDK] 成功加载 Caltech256 类别难度分域表: {domain_json_path}")
        else:
            print(f"[Warning] 找不到 {domain_json_path}，所有类别默认分配到 Domain 0")
            self.class_domain_map = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        # =======================================================
        # [DDK 新增] 直接根据类别标签获取 domain_id
        # =======================================================
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        # [DDK 修改] 返回三元组：图像，标签，领域ID
        return image, label, domain_id