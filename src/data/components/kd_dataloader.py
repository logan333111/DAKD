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
    def __init__(self, data_name, data_root, attributes, split='train', num_domains=10):
        self.root_dir = data_root
        self.split = split
        self.num_domains = num_domains or 10
        self.transform = self.get_transform(split)
        self.attributes = attributes #prompt_tmpl, class_num, classes
        self.dataloader = self.get_dataloader(data_name)

    def get_transform(self, split):
        normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])       
        if split == 'train':

            use_randaugment = True

            if use_randaugment:
                return transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.RandomCrop((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=2, magnitude=6),
                    transforms.ToTensor(),
                    normalize,
                    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
                    ])

            return transforms.Compose([
                transforms.Resize((256, 256)),
                #transforms.Pad(10),
                #transforms.Pad(20),
                transforms.RandomCrop((224, 224)),
                #transforms.CenterCrop((224, 224)),
                RandomCropAndResize(min_crop_size=192, max_crop_size=224, final_size=224),
                transforms.RandomRotation(60, interpolation = PIL.Image.BILINEAR),
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.00001, 10)),
                transforms.ColorJitter(brightness=.3, contrast=.1, saturation=.1, hue=.1),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
                #transforms.RandomErasing(p=0.5, scale=(0.3, 0.5), ratio=(0.3, 3.3),),
                ])
        else:
            return transforms.Compose([
                        # transforms.Scale(256),
                        # transforms.CenterCrop(224),
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        normalize,
                    ])

    def get_dataloader(self, data_name):
        if data_name == '0_CUB_200_2011':
            return CUB200Dataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '1_FGVC_AIRCRAFT':
            return AircraftDataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '2_NABirds':
            return NABirdsDataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '3_DTD':
            return DTDDataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '4_OxfordIIITPet':
            return OxfordIIPetDataset(self.root_dir, split=self.split, transform=self.transform, attributes=self.attributes, num_domains=self.num_domains)
        elif data_name == '5_StanfordDogs':
            return StanfordDogsDataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '6_StanfordCars':
            return StanfordCarsDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '7_CALTECH101':
            return Caltech101Dataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '8_CALTECH256':
            return Caltech256Dataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)
        elif data_name == '9_GTSRB':
            return GTSRBTestDataset(self.root_dir, split=self.split, transform=self.transform)
        elif data_name == '10_Oxford102Flowers':
            return Oxford102FlowersDataset(self.root_dir, split=self.split, transform=self.transform, num_domains=self.num_domains)

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
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split

        # Load train/test split
        splits = []
        with open(os.path.join(root_dir, 'train_test_split.txt'), 'r') as f:
            for line in f:
                _, is_train = line.strip().split(' ')
                splits.append(int(is_train))
        
        # Load image paths
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for idx, line in enumerate(f):
                img_id, img_path = line.strip().split(' ')
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.image_paths.append(os.path.join(root_dir, 'images', img_path))
                
        # Load class labels
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for idx, line in enumerate(f):
                _, label = line.strip().split(' ')
                if (split == "train" and splits[idx] == 1) or (split == "test" and splits[idx] == 0):
                    self.labels.append(int(label))
        
        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx] - 1  # Convert 1-based to 0-based
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id


class DTDDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
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
                # Use sorted() for deterministic class ordering
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

        # sorted() to align with YAML class ordering
        for label, category in enumerate(sorted(os.listdir(img_path))):
            category_path = os.path.join(img_path, category)
            if os.path.isdir(category_path):
                for img_name in sorted(os.listdir(category_path)):
                    if img_name.endswith('.jpg') and splits[img_name] == split:
                        self.image_paths.append(os.path.join(category_path, img_name))
                        self.labels.append(label)

        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        label = self.labels[idx]
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id


import os
import json
from PIL import Image
from torch.utils.data import Dataset

class AircraftDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
        assert split in ["train", "test", "val"], "split should be 'train', 'test', or 'val'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        # Build class name -> class_id mapping
        label_map = {}
        with open(os.path.join(root_dir, 'data', 'variants.txt'), 'r') as f:
            for idx, line in enumerate(f):
                class_name = line.strip()
                label_map[class_name] = idx

        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

        # Load image paths and labels from split file
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
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id

class NABirdsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        
        # Load image paths
        with open(os.path.join(root_dir, 'images.txt'), 'r') as f:
            for line in f:
                img_id, img_path = line.strip().split()
                self.image_paths.append(os.path.join(root_dir, 'images', img_path))

        # Load labels
        with open(os.path.join(root_dir, 'image_class_labels.txt'), 'r') as f:
            for line in f:
                _, label = line.strip().split()
                self.labels.append(int(label) - 1)  # Convert to 0-based
                
        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}
        
        # Filter by train/test split
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
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id

class OxfordIIPetDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, attributes=None, num_domains=10):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = self._build_class_to_idx(attributes)

        if split == "train":
            split_file = os.path.join(root_dir, "annotations", "trainval.txt")
        else:
            split_file = os.path.join(root_dir, "annotations", "test.txt")

        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

        # Load image paths and labels from split file
        with open(split_file, 'r') as file:
            for line in file:
                img_name, label_id, _, _ = line.strip().split()
                img_path = os.path.join(root_dir, "images", img_name + ".jpg")
                self.image_paths.append(img_path)

                class_name = self._normalize_class_name(img_name.rsplit('_', 1)[0])
                if class_name in self.class_to_idx:
                    self.labels.append(self.class_to_idx[class_name])
                else:
                    # Fallback to official 1-based label_id when attributes not provided
                    self.labels.append(int(label_id) - 1)
        
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _normalize_class_name(class_name):
        return str(class_name).replace('_', ' ').lower()

    def _build_class_to_idx(self, attributes):
        if attributes is None or not hasattr(attributes, 'classes'):
            return {}
        return {
            self._normalize_class_name(class_name): idx
            for idx, class_name in enumerate(attributes.classes.values())
        }

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id

class CalTech256Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # Generate split file if not exists
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
        
        # Load image paths and labels from split file
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
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        split_file = os.path.join(root_dir, "train_test_split.txt")
        
        # Generate split file if not exists
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
        
        # Load image paths and labels from split file
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
        assert split in ["train", "test"], "split should be either 'train' or 'test'"
        
        self.df = pd.read_csv(os.path.join(root_dir, f"{split}.csv"))
        self.img_dir = os.path.join(root_dir, "images")
        self.transform = transform
        self.domains = []
        
        # Load domain mapping (filename -> domain_id)
        domain_map = {}
        domain_file_path = os.path.join(root_dir, 'ddk_image_domain_labels.txt')
        if os.path.exists(domain_file_path):
            with open(domain_file_path, 'r') as f:
                for line in f:
                    filename, dom_id = line.strip().split()
                    domain_map[filename] = int(dom_id)
        else:
            print(f"[Warning] DDK domain file not found at {domain_file_path}. Defaulting domains to 0.")
        
        # Assign domain ID per image
        for idx in range(len(self.df)):
            filename = self.df.iloc[idx, 1]  # filename column
            self.domains.append(domain_map.get(filename, 0))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx, 1])  # filename column
        image = Image.open(img_name).convert('RGB')
        label = int(self.df.iloc[idx, 0])  # label column
        
        domain_id = self.domains[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id


class Oxford102FlowersDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
        assert split in ["train", "val", "test"], "split should be 'train', 'val', or 'test'"
        
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.split = split
        
        # Load labels from .mat file (1-based -> 0-based)
        labels_mat = scipy.io.loadmat(os.path.join(root_dir, 'imagelabels.mat'))
        all_labels = labels_mat['labels'][0] - 1  
        
        # Load split indices
        setid_mat = scipy.io.loadmat(os.path.join(root_dir, 'setid.mat'))
        if split == 'train':
            # Merge train and val for fine-grained tasks
            split_ids = setid_mat['trnid'][0].tolist() + setid_mat['valid'][0].tolist()
        elif split == 'val':
            split_ids = setid_mat['valid'][0].tolist()
        else: # test
            split_ids = setid_mat['tstid'][0].tolist()
            
        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}
        img_dir = os.path.join(root_dir, 'jpg')
        for img_id in split_ids:
            img_name = f"image_{img_id:05d}.jpg"
            self.image_paths.append(os.path.join(img_dir, img_name))
            
            self.labels.append(all_labels[img_id - 1])  # img_id is 1-based
            
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, domain_id

class StanfordDogsDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
        assert split in ["train", "test"], "split should be either 'train' or 'test'"

        self.split_mat = scipy.io.loadmat(os.path.join(root_dir, f"{split}_list.mat"))
        self.file_list = [item[0][0] for item in self.split_mat['file_list']]
        self.labels = [item[0] - 1 for item in self.split_mat['labels']]  # Convert 1-based index to 0-based
        
        self.img_dir = os.path.join(root_dir, "Images")
        self.transform = transform
        
        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.labels[idx]
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id


class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None):
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
    def __init__(self, root_dir, split="train", transform=None, num_domains=10):
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

        # Load class-to-domain mapping
        domain_json_path = os.path.join(root_dir, f'class_to_domain_{num_domains}.json')
        if os.path.exists(domain_json_path):
            import json
            with open(domain_json_path, 'r') as f:
                self.class_domain_map = json.load(f)
            if split == "train":
                print(f"[DDK] Loaded domain map: {domain_json_path}")
        else:
            print(f"[Warning] Domain map not found: {domain_json_path}, defaulting to domain 0")
            self.class_domain_map = {}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        domain_id = self.class_domain_map.get(str(label), 0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, domain_id
