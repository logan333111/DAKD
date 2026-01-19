#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torchvision.models as models
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

CKPT_PATH = "/data3/hgq/VL2Lite/VL2Lite-main/logs/train/runs/2025-12-15_17-07-00/checkpoints/epoch_299.ckpt"

def safe_load_ckpt(path: str):
    """
    先尝试安全的 weights_only=True；
    如果失败（你现在就是这种情况），自动降级到 weights_only=False。
    注意：weights_only=False 会反序列化对象，所以必须确保 ckpt 来源可信（你自己训练的就行）。
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        print("[Load] weights_only=True failed, fallback to weights_only=False", flush=True)
        print(f"[Load] reason: {type(e).__name__}: {e}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)

def find_best_prefix(state_dict: dict, target_keys: set):
    """
    自动在 ckpt 的 state_dict 里找“最像 resnet18 的那一坨参数”，返回最合适的前缀。
    """
    candidates = [
        "model.net.student.model.",
        "model.net.student.",
        "net.student.model.",
        "net.student.",
        "student.model.",
        "student.",
        "model.student.model.",
        "model.student.",
        "module.model.net.student.model.",
        "module.model.net.student.",
        "module.net.student.model.",
        "module.net.student.",
        "module.student.model.",
        "module.student.",
        "module.",
        "",
    ]

    best = ("", 0, 0)  # (prefix, matched, total)
    for p in candidates:
        sub = {k[len(p):]: v for k, v in state_dict.items() if k.startswith(p)}
        matched = len(set(sub.keys()) & target_keys)
        total = len(sub)
        if matched > best[1]:
            best = (p, matched, total)

    return best  # prefix, matched, total

def extract_with_prefix(state_dict: dict, prefix: str):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

def main():
    ckpt = safe_load_ckpt(CKPT_PATH)

    # Lightning ckpt 通常是 dict 且含 state_dict
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if not isinstance(sd, dict):
        raise ValueError("这个 ckpt 结构不对：没有可用的 state_dict")

    # 1) 先搭一个“裸 student：resnet18 -> 200类”
    student = models.resnet18(weights=None)
    student.fc = nn.Linear(student.fc.in_features, 200)
    target_keys = set(student.state_dict().keys())

    # 2) 自动找出 ckpt 里最像 resnet18 的前缀（通常就是 student 的那部分）
    prefix, matched, total = find_best_prefix(sd, target_keys)
    print(f"[Pick] best_prefix='{prefix}' matched={matched} / target={len(target_keys)} (sub_total={total})", flush=True)

    # 3) 抠出那部分参数并加载
    sub_sd = extract_with_prefix(sd, prefix)

    missing, unexpected = student.load_state_dict(sub_sd, strict=False)
    print(f"[Load] missing_keys={len(missing)} unexpected_keys={len(unexpected)}", flush=True)
    if missing:
        print("[Load] first_missing:", missing[:15], flush=True)
    if unexpected:
        print("[Load] first_unexpected:", unexpected[:15], flush=True)

    # 4) 做一次最简单的 forward 试跑（用随机输入，只验证能跑通）
    student.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = student(x)
        pred = logits.argmax(dim=1)
    print("[Forward] logits.shape:", tuple(logits.shape), "pred:", pred.tolist(), flush=True)

if __name__ == "__main__":
    main()
