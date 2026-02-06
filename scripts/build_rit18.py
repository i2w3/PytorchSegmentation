from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.io import loadmat


MAT_PATH = Path("datasets/rit18_data.mat")
SAVE_PATH = Path("datasets/rit18")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def calculate_statistics(image_data: np.ndarray):
    print("Calculating mean and std...")
    # image_data shape: (7, H, W)
    # Last channel is mask, first 6 are spectral bands
    bands = image_data[:6]
    mask = image_data[-1]

    # Get valid pixel mask
    valid_mask = mask != 0

    # Select valid pixels, shape: (6, N)
    valid_pixels = bands[:, valid_mask]

    means = np.mean(valid_pixels, axis=1)
    stds = np.std(valid_pixels, axis=1)

    print("Means:", means)
    print("Stds: ", stds)
    return means, stds


def generate_data(split: str, image_data:np.ndarray, label_data:np.ndarray, cut_size:int=512) -> None:
    assert split in ["train", "val", "test"], "Split must be 'train', 'val' or 'test'"
    assert image_data.shape[1:] == label_data.shape[:2], "Image and mask dimensions do not match"
    split_path = SAVE_PATH / split
    image_path = split_path / "images"
    label_path = split_path / "labels"
    split_path.mkdir(parents=True, exist_ok=True)
    image_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    mask_data  = image_data[-1]
    image_data = np.transpose(image_data[:6], (1, 2, 0))

    # 生成切片坐标：高度方向和宽度方向
    h_steps, w_steps = [], []
    c_h, c_w = 0, 0
    h, w = label_data.shape[:2]
    while c_h + cut_size <= h:
        h_steps.append(c_h)
        c_h += cut_size
    while c_w + cut_size <= w:
        w_steps.append(c_w)
        c_w += cut_size

    # 补充最后一个切片位置，确保覆盖到图像边界
    if len(h_steps) == 0 or h_steps[-1] != h - cut_size:
        h_steps.append(max(0, h - cut_size))
    if len(w_steps) == 0 or w_steps[-1] != w - cut_size:
        w_steps.append(max(0, w - cut_size))

    for idx_h, h_s in tqdm(enumerate(h_steps)):
        h_e = h_s + cut_size
        for idx_w, w_s in enumerate(w_steps):
            w_e = w_s + cut_size
            image_clip = image_data[h_s: h_e, w_s: w_e, :]
            label_clip  = label_data[h_s: h_e, w_s: w_e]
            mask_clip   = mask_data[h_s: h_e, w_s: w_e]

            image_clip[mask_clip == 0] = [0,0,0,0,0,0]
            label_clip[mask_clip == 0] = 0  # set invalid pixels to 0

            # save (512 * 512 * 6) image to tif
            image_save_path = image_path / f"{idx_h}_{idx_w}.tif"
            label_save_path  = label_path  / f"{idx_h}_{idx_w}.png"

            image_clip = np.transpose(image_clip, (2, 0, 1))  # to (C, H, W)

            tifffile.imwrite(str(image_save_path), image_clip)
            cv2.imwrite(str(label_save_path), label_clip)


if __name__ == "__main__":
    dataset = loadmat(MAT_PATH)

    #Load Training Data and Labels
    train_data = dataset['train_data']
    train_labels = dataset['train_labels']
    calculate_statistics(train_data)
    generate_data("train", train_data, train_labels)

    #Load Validation Data and Labels
    val_data = dataset['val_data']
    val_labels = dataset['val_labels']
    generate_data("val", val_data, val_labels)
    
    classes = dataset['classes']                          
    print("Classes:", classes)