'''
原始的 mask 标签说明：
    0: Ignore
    1: Background
    2: Building
    3: Road
    4: Water
    5: Barren
    6: Forest
    7: Agricultural
    8: Playground
修改后的 mask 标签说明：
    0: Background (包含原始的 Ignore 类别) 26.5%
    1: Building 7.3%
    2: Road 4.1%
    3: Water 9.7%
    4: Barren 11.6%
    5: Forest 28.9%
    6: Agricultural 11.9%
    7: Playground 0.1%
'''
from pathlib import Path
from typing import List
import shutil

import cv2
import numpy as np
from tqdm import tqdm


def change_name(path:Path, old_name:str, new_name:str) -> Path:
    # 将 TARGET_PATH 中的 OLD_NAME 替换为 NEW_NAME
    paris = path.parts
    if old_name in paris:
        new_parts = [new_name if part == old_name else part for part in paris]
        return Path(*new_parts)
    return path


def process(image_path:List[Path], save_path: Path) -> None:
    Path.mkdir(save_path / "images_png", parents=True, exist_ok=True)
    Path.mkdir(save_path / "masks_png", parents=True, exist_ok=True)
    for img_path in tqdm(image_path, desc=f"Copying images to {save_path}"):
        shutil.copy(img_path, save_path / "images_png")
        mask_path = change_name(img_path, "images_png", "masks_png")
        mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_data is None:
            print(f"Warning: mask_data is None for {mask_path}")
            continue
        # 修改 mask 标签，将标签从 1-8 变为 0-7
        for i in range(1, 9):
            mask_data[mask_data==i] = i-1
        cv2.imwrite(save_path / "masks_png" / img_path.name, mask_data)


if __name__ == "__main__":
    # 先下载数据集
    # from datasets import load_dataset

    # dataset_url = "Kingdrone-Junjue/EarthVLSet"
    # dataset = load_dataset(dataset_url)

    data_path = Path("~/.cache/huggingface/hub/datasets--Kingdrone-Junjue--EarthVLSet/snapshots/f75a6208eede63a97d29fb01a9b75800d73307ee/EarthVL-GLOBAL").expanduser()
    save_path = Path("./datasets/EarthVL")

    split = "train"
    save_path_split = save_path / split.capitalize()
    image_path = list((data_path / split.capitalize()).rglob("images_png/*.png"))
    process(image_path, save_path_split)

    split = "val"
    save_path_split = save_path / split.capitalize()
    image_path = list((data_path / split.capitalize()).rglob("images_png/*.png"))
    process(image_path, save_path_split)
    # 分析一下 val 集的 mask 分布
    unique_values = set()
    class_counts = {}
    for valid_sample in tqdm(list((save_path_split / "masks_png").rglob("*.png")), desc="Analyzing val masks"):
        mask  = cv2.imread(str(valid_sample), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupt: {valid_sample}")
        unique_value = np.unique(mask)
        for c in unique_value:
            int_c = int(c)
            class_counts[int_c] = class_counts.get(int_c, 0) + (mask == int_c).sum().item()
        unique_values.update(unique_value)
    print("Unique pixel values in validing mask images:", unique_values)
    total_pixels = sum(class_counts.values())
    class_freqs = {k: v / total_pixels for k, v in class_counts.items()}
    for k, v in sorted(class_freqs.items()):
        freq = 100.0 * v
        print(f"  Class {k}: {freq:5.1f}%")

    split = "test"
    save_path_split = save_path / split.capitalize()
    image_path = list((data_path / split.capitalize()).rglob("images_png/*.png"))
    Path.mkdir(save_path_split / "images_png", parents=True, exist_ok=True)
    for img_path in tqdm(image_path, desc="Copying test images"):
        shutil.copy(img_path, save_path_split / "images_png")
    