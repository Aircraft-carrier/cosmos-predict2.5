#!/usr/bin/env python3
"""
Inspect PyTorch .pt / .pth checkpoint file.
Supports:
  - ckpt (recommended)
  - full checkpoint dict (e.g., {'model': ..., 'optimizer': ..., 'epoch': ...})
  - pickled nn.Module (discouraged but handled)

Usage:
    python inspect_pt_checkpoint.py path/to/model.pt
"""

import torch
import argparse
import os
from collections import OrderedDict

def load_and_inspect_ckpt(ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"🔍 Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")  # Use CPU to avoid GPU issues
    print("\n" + "="*60)
    print("🧾 CHECKPOINT TYPE ANALYSIS")
    print()
    
    def safe_sum(tensor, name="tensor"):
        try:
            return tensor.abs().sum().item()
        except Exception as e:
            print(f"⚠️  Failed to sum {name}: {e}")
            return float('nan')
        
    RED = '\033[91m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    # 1. action_head.linear.weight (full only)
    w = ckpt['net.action_head.linear.weight']
    print(f"net.action_head.linear.weight (all)          : {safe_sum(w): .6f}")

    # 2. t_embedder.proj.linear_1.weight (split at 2048)
    w = ckpt['net.t_embedder.proj.linear_1.weight']
    print(f"net.t_embedder.proj.linear_1.weight (all)     : {safe_sum(w): .6f}")
    print(f"                                             {BLUE}[:, :2048]{RESET} : {safe_sum(w[:, :2048]): .6f}")
    print(f"                                             {RED}[:, 2048:]{RESET} : {safe_sum(w[:, 2048:]): .6f}")

    # 3. adaln_modulation_self_attn.1.weight
    w = ckpt['net.blocks.0.adaln_modulation_self_attn.1.weight']
    print(f"net.blocks.0.adaln_modulation_self_attn.1.weight (all)  : {safe_sum(w): .6f}")
    print(f"                                             {BLUE}[:, :2048]{RESET} : {safe_sum(w[:, :2048]): .6f}")
    print(f"                                             {RED}[:, 2048:]{RESET} : {safe_sum(w[:, 2048:]): .6f}")

    # 4. adaln_modulation_cross_attn.1.weight
    w = ckpt['net.blocks.0.adaln_modulation_cross_attn.1.weight']
    print(f"net.blocks.0.adaln_modulation_cross_attn.1.weight (all) : {safe_sum(w): .6f}")
    print(f"                                             {BLUE}[:, :2048]{RESET} : {safe_sum(w[:, :2048]): .6f}")
    print(f"                                             {RED}[:, 2048:]{RESET} : {safe_sum(w[:, 2048:]): .6f}")

    # 5. adaln_modulation_mlp.1.weight
    w = ckpt['net.blocks.0.adaln_modulation_mlp.1.weight']
    print(f"net.blocks.0.adaln_modulation_mlp.1.weight (all)        : {safe_sum(w): .6f}")
    print(f"                                             {BLUE}[:, :2048]{RESET} : {safe_sum(w[:, :2048]): .6f}")
    print(f"                                             {RED}[:, 2048:]{RESET} : {safe_sum(w[:, 2048:]): .6f}")

    # 6. t_embedding_norm.weight (1D tensor)
    w = ckpt['net.t_embedding_norm.weight']
    print(f"net.t_embedding_norm.weight (all)                      : {safe_sum(w): .6f}")
    print(f"                                             {BLUE}[:2048]{RESET}    : {safe_sum(w[:2048]): .6f}")
    print(f"                                             {RED}[2048:]{RESET}    : {safe_sum(w[2048:]): .6f}")

    # 7. action_head.adaln_modulation.1.weight (full only)
    w = ckpt['net.action_head.adaln_modulation.1.weight']
    print(f"net.action_head.adaln_modulation.1.weight (all)         : {safe_sum(w): .6f}")

    # 8. action_embedder.fc2.weight (full only)
    w = ckpt['net.action_embedder.fc2.weight']
    print(f"net.action_embedder.fc2.weight (all)                    : {safe_sum(w): .6f}")

    print("\n" + "="*60)
    print("✅ Inspection completed.")
    return ckpt

def _print_ckpt_info(ckpt):
    print(f"\n📊 Model ckpt summary:")
    print(f"  Total parameters: {len(ckpt)} layers")
    
    total_params = 0
    for name, param in ckpt.items():
        num_params = param.numel()
        total_params += num_params
        print(f"  - {name}: {list(param.shape)} ({num_params:,} params)")
    
    print(f"\n📈 Total trainable parameters: {total_params:,}")
    if total_params >= 1e9:
        print(f"   ≈ {total_params / 1e9:.2f} B")
    elif total_params >= 1e6:
        print(f"   ≈ {total_params / 1e6:.2f} M")

def main():
    parser = argparse.ArgumentParser(description="Inspect PyTorch checkpoint file")
    parser.add_argument("--ckpt_path", type=str, help="Path to .pt or .pth file")
    args = parser.parse_args()

    try:
        load_and_inspect_ckpt(args.ckpt_path)
    except Exception as e:
        print(f"💥 Error loading checkpoint: {e}")
        raise

if __name__ == "__main__":
    main()