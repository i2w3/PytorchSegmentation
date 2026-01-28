from collections import OrderedDict
from typing import Union
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import kagglehub
import numpy as np
from torch.utils.data import Dataset, DataLoader


def change_name(path:Path, old_name:str, new_name:str) -> Path:
    # 将 TARGET_PATH 中的 OLD_NAME 替换为 NEW_NAME
    paris = path.parts
    if old_name in paris:
        new_parts = [new_name if part == old_name else part for part in paris]
        return Path(*new_parts)
    return path

def get_train_transforms():
    return A.Compose([A.RandomResizedCrop(size=[512, 512], scale=(0.6, 1.0), ratio=(0.9, 1.1),
                                          interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.RandomRotate90(p=0.5),
                      A.Affine(scale=1.0, translate_percent=0.05, rotate=15,
                               interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST, p=0.5),
                      A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                      ToTensorV2(),
    ],strict=True)

def get_other_transforms():
    return A.Compose([A.Resize(height=512, width=512, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                      A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                      ToTensorV2(),
    ],strict=True)

def make_loader(dataset, batch_size, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True if shuffle else False,
    )

    
class EarthVQA(Dataset):
    LABEL_MAP = OrderedDict(Background=0,
                            Building=1,
                            Road=2,
                            Water=3,
                            Barren=4,
                            Forest=5,
                            Agricultural=6,
                            Playground=7,
                            Pond=8,
                )
    def __init__(self, data_path:Union[str, Path] = "./datasets/EarthVQA", split:str = "train",augmentation=None, num_classes:int=9):
        super().__init__()
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path
        self.augmentation = augmentation
        self.num_classes = num_classes
        self.image_paths = list((self.data_path / split.capitalize()).rglob("images_png/*.png"))
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = change_name(img_path, "images_png", "masks_png")
        
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or corrupt: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found or corrupt: {mask_path}")
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
        return image, mask


if __name__ == "__main__":
    import numpy as np
    from tqdm import tqdm

    train_transform = get_train_transforms()
    valid_transform = get_other_transforms()
    
    train_dataset = EarthVQA(split="train", augmentation=train_transform)
    valid_dataset = EarthVQA(split="val",   augmentation=valid_transform)
    print(f"len of train-valid is {len(train_dataset)}-{len(valid_dataset)}")

    unique_values = set()
    class_counts = {}
    for train_sample in tqdm(train_dataset):
        image, mask = train_sample
        image, mask = image.numpy(), mask.numpy()
        unique_value = np.unique(mask)
        for c in unique_value:
            int_c = int(c)
            class_counts[int_c] = class_counts.get(int_c, 0) + (mask == int_c).sum().item()
        unique_values.update(unique_value)
    print("Unique pixel values in training mask images:", unique_values)
    total_pixels = sum(class_counts.values())
    class_freqs = {k: v / total_pixels for k, v in class_counts.items()}
    for k, v in sorted(class_freqs.items()):
        freq = 100.0 * v
        print(f"  Class {k}: {freq:5.1f}%")

    unique_values = set()
    class_counts = {}
    for valid_sample in tqdm(valid_dataset):
        image, mask = valid_sample
        image, mask = image.numpy(), mask.numpy()
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
