import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.set_float32_matmul_precision('high')

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import lightly_train
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np

from utils import change_name

# MODEL = "dinov3/vitt16-eomt-coco"
MODEL = "dinov3/vits16-eomt-coco"

CONFIG = {
    "classes_name": ["Background", "Building", "Road", "Water", "Barren", "Forest", "Agricultural", "Playground", "Pond"],
    "classes_name_zh": ["背景", "建筑", "道路", "水", "裸地", "森林", "农业", "操场", "池塘"],
    "classes_cmap": [[0, 0, 0], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128], [165, 0, 165], [0, 255, 255]],
}


def parse_args():
    parser = argparse.ArgumentParser(description="EoMT 语义分割预训练、微调与导出脚本")
    parser.add_argument("task", type=int, help="任务名称")
    parser.add_argument("--ids", type=int, default=0, help="用于可视化的样本 ID")
    return parser.parse_args()


def plot_data(config:dict, 
              img_data:np.ndarray,
              mask_data:np.ndarray, 
              pred_data:np.ndarray,
              fig:Optional[plt.Figure] = None, 
              save_str:Optional[str] = None) -> plt.Figure:
    '''绘制图像数据和预测结果
    '''
    if fig is None:
        fig = plt.figure(figsize=(15, 10))
    fig.clf() # 清除之前的内容
    axes = fig.subplots(2, 2)
    ## Row 1
    axes[0,0].imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    axes[0,0].set_title("img")
    axes[0,0].axis('off')
    legend_elements = []
    for i, class_name in enumerate(config["classes_name"]):
        color = np.array(config["classes_cmap"][i]) / 255.0
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', label=class_name, markerfacecolor=color, markersize=10))
    axes[0,1].axis('off')
    axes[0,1].legend(handles=legend_elements, loc='center', fontsize='large')
    ## Row 2
    mask, img1 = over_leap(config, img_data, mask_data)
    pred, img2 = over_leap(config, img_data, pred_data)

    axes[1,0].imshow(img1)
    axes[1,0].set_title("mask")
    axes[1,0].axis('off')
    axes[1,1].imshow(img2)
    axes[1,1].set_title("pred")
    axes[1,1].axis('off')

    plt.tight_layout()
    if save_str is not None:
        plt.savefig(save_str)
    return fig


def over_leap(config:dict, img_data:np.ndarray, mask_data:np.ndarray) -> List[np.ndarray]:
    '''图像叠加
    '''
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    if img.shape[:2] != mask_data.shape[:2]:
        mask_data = cv2.resize(mask_data, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    # 将 mask 不为 0 的区域，使用 config["classes_cmap"] 中对应的颜色进行替换
    color_mask = np.zeros_like(img)
    for i, color in enumerate(config["classes_cmap"]):
        color_mask[mask_data == i] = color
    # 将 mask_data 不为 0 的区域，使用 255 替换
    mask_data_bin = np.where(mask_data <= 1, 0, 255).astype(np.uint8)
    return [mask_data_bin,color_mask]


if __name__ == "__main__":
    args = parse_args()
    data_path = Path("./datasets/EarthVQA")
    if args.task == 1:
        # 1. Pretrain the test split
        test_path = data_path / "Test" / "images_png"
        lightly_train.pretrain(
            out=f"logs/{Path(MODEL).stem}_pretrain",
            data=test_path,
            model=MODEL.split("-")[0],
            method="dinov2",
            batch_size=256,
            num_workers=16,
            devices=[1,2],
            resume_interrupted=True, # 不知道为何容易一张显卡上的进程丢失，建议加上这个
        )
    elif args.task == 2:
        # 2. Fine-tune on the training split
        lightly_train.train_semantic_segmentation(
            out=f"logs/{Path(MODEL).stem}",
            model=MODEL,
            model_args={
                # Path to your pretrained DINOv2 model.
                "backbone_weights": f"logs/{Path(MODEL).stem}_pretrain/exported_models/exported_last.pt",
            },
            data={
                "train": {
                    "images": f"{data_path}/Train/images_png",   # Path to training images
                    "masks": f"{data_path}/Train/masks_png",     # Path to training masks
                },
                "val": {
                    "images": f"{data_path}/Val/images_png",     # Path to validation images
                    "masks": f"{data_path}/Val/masks_png",       # Path to validation masks
                },
                "classes": { 
                    0: "Background",
                    1: "Building",
                    2: "Road",
                    3: "Water",
                    4: "Barren",
                    5: "Forest",
                    6: "Agricultural",
                    7: "Playground",
                    # 8: "Pond",
                },
                # "ignore_classes": [0], 
            },
            transform_args={
                "image_size": (512, 512),
                "normalize": {
                    "mean": [0.485, 0.456, 0.406],
                    "std": [0.229, 0.224, 0.225],
                },
                "val": {
                    "image_size": (512, 512),
                    "normalize": {
                        "mean": [0.485, 0.456, 0.406],
                        "std": [0.229, 0.224, 0.225],
                    },
                },
            },
            batch_size=32,
            num_workers=16,
            devices=[4,5],
            overwrite=True,
        )
    elif args.task > 2:
        model = lightly_train.load_model(f"logs/{Path(MODEL).stem}/exported_models/exported_best.pt")
        if args.task == 3:
            print(f"Visualizing example ids[{args.ids}]...")
            image_paths = list((data_path / "Val").rglob("images_png/*.png"))
            image_data = read_image(str(image_paths[args.ids]))
            mask_path = change_name(image_paths[args.ids], "images_png", "masks_png")
            mask_data = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            pred_data = model.predict(image_data)
            plot_data(CONFIG, image_data.permute(1,2,0).numpy(), mask_data, pred_data.cpu().numpy(), save_str=f"./assets/example_plot_{args.ids}.png")
        elif args.task == 4:
            print("Exporting the model to ONNX format...")
            model.export_onnx(
                out=f"logs/{Path(MODEL).stem}/exported_models/{Path(MODEL).stem}_EarthVQA.onnx",
                precision="fp32", # Export model with FP16 weights for smaller size and faster inference.
                simplify=False,  # Simplify the model using onnx-simplifier.
                batch_size=1,
                height=512,
                width=512,
            )
        else:
            raise ValueError("Invalid task number.")
    else:
        raise ValueError("Invalid task number.")