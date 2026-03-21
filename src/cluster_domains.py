import os
import torch
import numpy as np
import scipy.io
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
import torchvision.transforms as transforms
import torchvision.models as models

def main():
    root_dir = '/data2/hgq/VL2Lite/VL2Lite-main/data/kd_datasets/5_StanfordDogs'
    img_dir = os.path.join(root_dir, 'Images')
    
    # 1. Load train and test lists
    print("Loading file lists...")
    train_mat = scipy.io.loadmat(os.path.join(root_dir, 'train_list.mat'))
    test_mat = scipy.io.loadmat(os.path.join(root_dir, 'test_list.mat'))
    
    train_files = [item[0][0] for item in train_mat['file_list']]
    test_files = [item[0][0] for item in test_mat['file_list']]
    all_files = train_files + test_files
    print(f"Total images to process: {len(all_files)}")
    
    # 2. Setup Feature Extractor (ResNet50 / ViT)
    print("Loading pretrained feature extractor...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Identity()  # Remove classification head to get raw features (2048-dim)
    model = model.to(device)
    model.eval()
    
    # Standard ImageNet Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Extract Features
    bs = 128
    all_features = []
    
    print("Extracting features...")
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), bs)):
            batch_files = all_files[i:i+bs]
            batch_imgs = []
            valid_files = []
            
            for f in batch_files:
                path = os.path.join(img_dir, f)
                try:
                    img = Image.open(path).convert('RGB')
                    batch_imgs.append(transform(img))
                    valid_files.append(f)
                except Exception as e:
                    print(f"Error loading {path}: {e}")
                    
            if not batch_imgs:
                continue
                
            inputs = torch.stack(batch_imgs).to(device)
            features = model(inputs)
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            all_features.append(features.cpu().numpy())
            
    all_features = np.concatenate(all_features, axis=0)
    print(f"Extracted feature shape: {all_features.shape}")
    
    # 4. K-Means Clustering into 10 domains
    num_domains = 10
    print(f"Clustering into {num_domains} domains using K-Means...")
    kmeans = KMeans(n_clusters=num_domains, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(all_features)
    
    # 5. Save to ddk_image_domain_labels.txt
    output_file = os.path.join(root_dir, 'ddk_image_domain_labels_kmeans.txt')
    print(f"Saving new domain mapping to {output_file}...")
    
    with open(output_file, 'w') as f:
        for filename, domain_id in zip(all_files, cluster_labels):
            f.write(f"{filename} {domain_id}\n")
            
    print("Done! You can rename this file to 'ddk_image_domain_labels.txt' and train again.")

if __name__ == "__main__":
    main()
