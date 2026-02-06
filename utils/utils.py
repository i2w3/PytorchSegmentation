from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch

def change_name(path:Path, old_name:str, new_name:str) -> Path:
    # 将 TARGET_PATH 中的 OLD_NAME 替换为 NEW_NAME
    paris = path.parts
    if old_name in paris:
        new_parts = [new_name if part == old_name else part for part in paris]
        return Path(*new_parts)
    return path


def view_rit18(sample:dict[str, torch.Tensor], pred:torch.Tensor) -> Figure:
    assert sample['image'].dim() == 3   # (C, H, W)
    assert sample['mask'].dim() == 2    # (H, W)
    assert pred.dim() == 2              # (H, W)
    figure = plt.figure(figsize=(10, 5))
    rgb_img = sample['image'][[3, 4, 5], :, :, ]
    rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy().astype('uint16')
    rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)

    mask_np = sample['mask'].cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    plt.subplot(1, 3, 1)
    plt.title("False Color RGB (Bands 4,5,6)")
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask_np, cmap='tab20', vmin=0, vmax=19)
    plt.title(f"Ground Truth Mask with {np.unique(mask_np)}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap='tab20', vmin=0, vmax=19)
    plt.title(f"Predicted Mask with {np.unique(pred_np)}")
    plt.axis('off')

    plt.tight_layout()
    return figure