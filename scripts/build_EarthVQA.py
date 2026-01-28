'''
从 kaggle 下载 EarthVQA 数据集，并进行预处理，生成适合训练和验证的文件结构，注意：修改了 mask 标签
数据集链接：https://www.kaggle.com/datasets/dyiyacao/earthvqa
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
    9: Pond # 这个最好合并到 Water 类别中
修改后的 mask 标签说明：
    0: Background (包含原始的 Ignore 类别) 34.0%
    1: Building 7.0%
    2: Road 4.6%
    3: Water (包含 Pond 类别) 12.0%
    4: Barren 4.3%
    5: Forest 7.5%
    6: Agricultural 30.5%
    7: Playground 0.1%
'''
from pathlib import Path
from typing import List
import shutil

import cv2
import kagglehub
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
        # 修改 mask 标签，将标签从 1-9 变为 0-8，并将 pond 类别（8）改为 water 类别（3）
        for i in range(1, 10):
            mask_data[mask_data==i] = i-1
        mask_data[mask_data==8] = 3 # 将 pond 类别改为 water 类别
        cv2.imwrite(save_path / "masks_png" / img_path.name, mask_data)


if __name__ == "__main__":
    data_path = Path(kagglehub.dataset_download("dyiyacao/earthvqa"))
    save_path = Path("./datasets/EarthVQA")

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
    image_path = list((data_path / "images_png").rglob("*.png"))
    Path.mkdir(save_path_split / "images_png", parents=True, exist_ok=True)
    for img_path in tqdm(image_path, desc="Copying test images"):
        shutil.copy(img_path, save_path_split / "images_png")
    