#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os
import random
from PIL import Image
from torchvision import transforms
import PIL
import torch
from torch.utils.data import DataLoader
# ===== Step4: 模拟训练：forward 一次 =====
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import argparse
import open_clip
import numpy as np
from types import SimpleNamespace
class CUB200Raw:
    """
    只做两件事：
    1) 解析 split，得到 image_paths / labels
    2) __getitem__ 返回 (PIL.Image, label_int)
    """
    def __init__(self, root_dir: str, split: str = "train"):
        assert split in ["train", "test"]
        self.root_dir = root_dir
        self.split = split
        self.image_paths = []
        self.labels = []
        self.transform = build_transform(split)

        split_file = os.path.join(root_dir, "train_test_split.txt")
        images_file = os.path.join(root_dir, "images.txt")
        labels_file = os.path.join(root_dir, "image_class_labels.txt")

        for p in [split_file, images_file, labels_file]:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"缺少文件: {p}")

        # 1) 读 train/test 标记
        splits = []
        with open(split_file, "r") as f:
            for line in f:
                _, is_train = line.strip().split()
                splits.append(int(is_train))

        # 2) 读图片相对路径 -> 拼成绝对路径
        with open(images_file, "r") as f:
            for idx, line in enumerate(f):
                _, rel_path = line.strip().split()
                is_train = splits[idx]
                if (split == "train" and is_train == 1) or (split == "test" and is_train == 0):
                    self.image_paths.append(os.path.join(root_dir, "images", rel_path))

        # 3) 读 label（原始是 1-based）
        with open(labels_file, "r") as f:
            for idx, line in enumerate(f):
                _, label = line.strip().split()
                is_train = splits[idx]
                if (split == "train" and is_train == 1) or (split == "test" and is_train == 0):
                    self.labels.append(int(label) - 1)  # 转成 0-based

        if len(self.image_paths) != len(self.labels):
            raise RuntimeError(f"paths/labels 数量不一致: {len(self.image_paths)} vs {len(self.labels)}")

        #print(f"[Step1] split={split} num_samples={len(self.image_paths)}")
        #print(f"[Step1] example_path={self.image_paths[0]}")
        #print(f"[Step1] example_label={self.labels[0]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"图片不存在: {img_path}")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)   # PIL -> Tensor[3,224,224]
        label = self.labels[idx]
        return img, label

def load_student_from_kd_ckpt(student: torch.nn.Module, ckpt_path: str,
                              prefix: str = "net.student.model.resnet."):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # 关键：prefix 直接用 net.student.model.resnet.
    sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

    missing, unexpected = student.load_state_dict(sub_sd, strict=False)
    print(f"[LoadStudent] prefix={prefix} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("[LoadStudent] first_missing:", missing[:10], flush=True)
    if unexpected:
        print("[LoadStudent] first_unexpected:", unexpected[:10], flush=True)
def load_head_from_kd_ckpt(head: torch.nn.Module, ckpt_path: str,
                           prefix: str = "net.student.model.linear_cls."):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    sub_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = head.load_state_dict(sub_sd, strict=False)
    print(f"[LoadHead] prefix={prefix} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("[LoadHead] first_missing:", missing[:10], flush=True)
    if unexpected:
        print("[LoadHead] first_unexpected:", unexpected[:10], flush=True)


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

def build_transform(split):
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            RandomCropAndResize(min_crop_size=192, max_crop_size=224, final_size=224),
            transforms.RandomRotation(60, interpolation=PIL.Image.BILINEAR),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.00001, 10)),
            transforms.ColorJitter(brightness=.3, contrast=.1, saturation=.1, hue=.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


def main():
    # 你把这里改成你的真实路径
    root = "/data3/hgq/VL2Lite/VL2Lite-main/data/kd_datasets/0_CUB_200_2011"
    ds = CUB200Raw(root_dir=root, split="train")

    # ===== 0) 参数 =====
    teacher_arch = "convnext_xxlarge"
    teacher_pretrained = "/data3/hgq/VL2Lite/VL2Lite-main/src/models/teacher/open_clip_pytorch_model.bin"
    prompt_tmpl = "A photo of a {}"

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="你的 VL2Lite Lightning ckpt 路径")
    parser.add_argument("--teacher_ckpt", default=teacher_pretrained, help="open_clip teacher 权重路径")
    parser.add_argument("--T", type=float, default=2.0, help="蒸馏温度 T")
    args = parser.parse_args()

    T = float(args.T)

    # ===== Step3: DataLoader 组 batch =====
    loader = DataLoader(
        ds,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    def feature_norm(x):
        return x / (x.norm(dim=-1, keepdim=True) + 1e-10)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Step4: 学生 = backbone(resnet.fc=Identity) + linear_cls（方式1）=====
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()  # 关键：不用 resnet 的 fc

    linear_cls = nn.Linear(512, 200)

    # 加载 ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # 1) 加载 backbone（对应 net.student.model.resnet.*）
    prefix_backbone = "net.student.model.resnet."
    sub_backbone = {k[len(prefix_backbone):]: v for k, v in sd.items() if k.startswith(prefix_backbone)}
    msg = backbone.load_state_dict(sub_backbone, strict=False)
    print(f"[LoadBackbone] prefix={prefix_backbone} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
    if msg.missing_keys:
        print("[LoadBackbone] first_missing:", msg.missing_keys[:10], flush=True)
    if msg.unexpected_keys:
        print("[LoadBackbone] first_unexpected:", msg.unexpected_keys[:10], flush=True)

    # 2) 加载 linear_cls（对应 net.student.model.linear_cls.*）
    prefix_head = "net.student.model.linear_cls."
    sub_head = {k[len(prefix_head):]: v for k, v in sd.items() if k.startswith(prefix_head)}
    msg2 = linear_cls.load_state_dict(sub_head, strict=False)
    print(f"[LoadHead] prefix={prefix_head} missing={len(msg2.missing_keys)} unexpected={len(msg2.unexpected_keys)}", flush=True)
    if msg2.missing_keys:
        print("[LoadHead] first_missing:", msg2.missing_keys[:10], flush=True)
    if msg2.unexpected_keys:
        print("[LoadHead] first_unexpected:", msg2.unexpected_keys[:10], flush=True)

    backbone = backbone.to(device).eval()
    linear_cls = linear_cls.to(device).eval()

    # ===== Step4.5: open_clip teacher（冻结）=====
    teacher_model, _, _ = open_clip.create_model_and_transforms(teacher_arch, pretrained=args.teacher_ckpt)
    teacher_model = teacher_model.to(device).eval()
    teacher_model.requires_grad_(False)
    tokenizer = open_clip.get_tokenizer(teacher_arch)

    # ===== Step5: 读取 200 类名 =====
    classes_txt = "/data3/hgq/VL2Lite/VL2Lite-main/data/kd_datasets/0_CUB_200_2011/classes.txt"
    class_names = []
    with open(classes_txt, "r") as f:
        for line in f:
            _id, name = line.strip().split(" ", 1)
            name = name.split(".", 1)[-1].replace("_", " ")
            class_names.append(name)
    assert len(class_names) == 200

    # ===== Step6: 冻结文本特征 clip_nlp_features（一次算好）=====
    prompts = [prompt_tmpl.format(n) for n in class_names]
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        clip_nlp_features = feature_norm(teacher_model.encode_text(text_tokens))  # [200,1024]

    # ===== Step7: 取一个 batch =====
    xb, yb = next(iter(loader))
    xb = xb.to(device)
    yb = yb.to(device)

    with torch.no_grad():
        # 学生特征 + 分类
        feats = backbone(xb)                       # [B,512]  (因为 fc=Identity)
        hidden_features = feature_norm(feats)      # [B,512]
        student_logits = linear_cls(feats)         # [B,200]

        # teacher 图像特征（open_clip 输出 1024）
        clip_img_features = feature_norm(teacher_model.encode_image(xb))  # [B,1024]

    # ===== Step8: AlignNet（1024->512），优先从 ckpt 加载；没有就随机初始化 =====
    align_img_layer = nn.Sequential(
        nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512)
    ).to(device).eval()
    align_nlp_layer = nn.Sequential(
        nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, 512)
    ).to(device).eval()

    def load_by_prefix(module, prefix):
        sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        msg = module.load_state_dict(sub, strict=False)
        print(f"[LoadAlign] prefix={prefix} missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}", flush=True)
        if msg.missing_keys:
            print("[LoadAlign] first_missing:", msg.missing_keys[:10], flush=True)

    # 只在 ckpt 里有 align 的情况下能 missing=0；没有就 missing=4（正常）
    load_by_prefix(align_img_layer, "net.align.align_img_layer.")
    load_by_prefix(align_nlp_layer, "net.align.align_nlp_layer.")

    with torch.no_grad():
        aligned_img = feature_norm(align_img_layer(clip_img_features))   # [B,512]
        aligned_nlp = feature_norm(align_nlp_layer(clip_nlp_features))   # [200,512]

    # ===== Step9: 打印预测 vs 真实 =====
    pred = student_logits.argmax(dim=1)
    print("[Step4] pred:", pred.detach().cpu().tolist(), flush=True)
    print("[Step4] true:", yb.detach().cpu().tolist(), flush=True)

    # ===== Step10: 三个 loss（L_cls + L_vis + L_txt）=====
    L_cls = F.cross_entropy(student_logits, yb)
    L_vis = F.smooth_l1_loss(hidden_features, aligned_img)

    logit_scale = float(np.exp(np.log(1 / 0.07)))
    student_nlp_logits = logit_scale * (hidden_features @ aligned_nlp.T) / T      # [B,200]
    teacher_nlp_logits = logit_scale * (clip_img_features @ clip_nlp_features.T) / T  # [B,200]

    # ====== [DEBUG] 直观查看：teacher vs student 与文本特征的相似度计算 ======
    def topk_print(title, logits, class_names, yb, k=5, max_show=3):
        probs = F.softmax(logits, dim=1)
        topv, topi = probs.topk(k, dim=1)
        print(f"\n===== {title} (Top-{k}) =====")
        for b in range(min(max_show, logits.size(0))):
            gt = int(yb[b].item())
            gt_prob = float(probs[b, gt].item())
            gt_rank = int((probs[b] > probs[b, gt]).sum().item()) + 1
            pairs = [(int(i), class_names[int(i)], float(v)) for v, i in zip(topv[b], topi[b])]
            print(f"[sample {b}] GT={gt} ({class_names[gt]})  GT_prob={gt_prob:.4f}  GT_rank={gt_rank}")
            print("  " + " | ".join([f"{idx}:{name}:{p:.4f}" for idx, name, p in pairs]))

    topk_print("Teacher: clip_img_features @ clip_nlp_features.T", teacher_nlp_logits, class_names, yb, k=5)
    topk_print("Student: hidden_features @ aligned_nlp.T", student_nlp_logits, class_names, yb, k=5)

    t_pred = teacher_nlp_logits.argmax(dim=1)
    s_pred = student_nlp_logits.argmax(dim=1)
    agree = (t_pred == s_pred).float().mean().item()
    print("\n[Compare] teacher_top1:", t_pred[:8].detach().cpu().tolist())
    print("[Compare] student_top1:", s_pred[:8].detach().cpu().tolist())
    print(f"[Compare] top1_agree={agree:.4f}")

    t_prob = F.softmax(teacher_nlp_logits, dim=1)
    s_logprob = F.log_softmax(student_nlp_logits, dim=1)
    kl_debug = F.kl_div(s_logprob, t_prob, reduction="batchmean") * (T * T)
    print(f"[Compare] KL(student||teacher)={kl_debug.item():.6f}")
    # ====== [DEBUG] end ======

    L_txt = F.kl_div(
        F.log_softmax(student_nlp_logits, dim=1),
        F.softmax(teacher_nlp_logits, dim=1),
        reduction="batchmean",
    ) * (T * T)

    # ===== Step11: 总 loss =====
    lam_cls, lam_vis, lam_txt = 0.01, 0.495, 0.495
    L_total = lam_cls * L_cls + lam_vis * L_vis + lam_txt * L_txt

    acc = (pred == yb).float().mean().item()
    print(f"[Loss] cls={L_cls.item():.6f} vis={L_vis.item():.6f} txt={L_txt.item():.6f} total={L_total.item():.6f} acc={acc:.4f}", flush=True)


if __name__ == "__main__":
 main()
 