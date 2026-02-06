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
    classes = {
        0: "Other Class/Image Border",
        1: "Road Markings",
        2: "Tree",
        3: "Building",
        4: "Vehicle (Car, Truck, or Bus)",
        5: "Person",
        6: "Lifeguard Chair",
        7: "Picnic Table",
        8: "Black Wood Panel",
        9: "White Wood Panel",
        10: "Orange Landing Pad",
        11: "Water Buoy",
        12: "Rocks",
        13: "Other Vegetation",
        14: "Grass",
        15: "Sand",
        16: "Water (Lake)",
        17: "Water (Pond)",
        18: "Asphalt (Parking Lot/Walkway)",
    }
    assert sample['image'].dim() == 3   # (C, H, W)
    assert sample['mask'].dim() == 2    # (H, W)
    assert pred.dim() == 2              # (H, W)
    figure = plt.figure(figsize=(15, 10))
    rgb_img = sample['image'][[3, 4, 5], :, :, ]
    rgb_img = rgb_img.permute(1, 2, 0).cpu().numpy().astype('uint16')
    rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)

    mask_np = sample['mask'].cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    plt.subplot(2, 2, 1)
    plt.title("False Color RGB (Bands 4,5,6)")
    plt.imshow(rgb_img)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=classes[i], markerfacecolor=plt.cm.tab20(i / 19), markersize=10) for i in range(19)]
    plt.axis('off')
    plt.legend(handles=legend_elements, loc='center', fontsize='large')

    plt.subplot(2, 2, 3)
    plt.imshow(mask_np, cmap='tab20', vmin=0, vmax=18)
    plt.title(f"Ground Truth Mask with {np.unique(mask_np)}")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(pred_np, cmap='tab20', vmin=0, vmax=18)
    plt.title(f"Predicted Mask with {np.unique(pred_np)}")
    plt.axis('off')

    plt.tight_layout()
    return figure