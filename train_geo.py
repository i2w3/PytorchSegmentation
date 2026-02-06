import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.set_float32_matmul_precision('high')

from pathlib import Path
from typing import Callable

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import tifffile
from torchvision.io import decode_image as load_image
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.datasets import NonGeoDataset
from torchgeo.trainers import SemanticSegmentationTask

from utils import change_name, view_rit18, plt


class RIT18(NonGeoDataset):
    '''
    无地理信息的 torchgeo 数据集封装类示例：RIT-18 数据集
    '''
    def __init__(self, 
                 root: Path = './datasets/rit18', 
                 split: str = 'train', 
                 transforms: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None,) -> None:
        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.split = split
        self.transforms = transforms
        self.tif_paths = list((self.root / split / "images").glob("*.tif"))

    def __len__(self) -> int:
        return len(self.tif_paths)
    
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tif_path = self.tif_paths[index]
        mask_path = change_name(tif_path, "images", "labels").with_suffix(".png")

        image = torch.from_numpy(tifffile.imread(str(tif_path)))
        mask = load_image(str(mask_path), mode="GRAY") # (1, H, W) 后面变成 (H, W)

        sample: dict[str, torch.Tensor] = {'image': image.float(), 'mask': mask.squeeze(0).long()}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
    

if __name__ == "__main__":
    datamodule = NonGeoDataModule(
        dataset_class=RIT18,
        root="./datasets/rit18",
        batch_size=16,
        num_workers=4,
        transforms=None, # TODO: 可以添加数据增强
    )

    task = SemanticSegmentationTask(model="unet", backbone="resnet50", weights=True, in_channels=6, num_classes=19, loss="jaccard")

    checkpoint_callback = ModelCheckpoint(
        monitor='val_MulticlassJaccardIndex', 
        mode='max',
        dirpath="./logs/torcgeo_rit18", 
        save_top_k=1, 
        save_last=True, 
        save_weights_only=True, 
    )
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10)
    logger = [CSVLogger("logs", name="torcgeo_rit18")]

    trainer = Trainer(accelerator="gpu", 
                      devices=[0,1], 
                      default_root_dir="./logs/torcgeo_rit18",
                      callbacks=[checkpoint_callback, early_stopping_callback],
                      max_epochs = 50,
                      logger=logger,
                      )

    trainer.fit(model=task, datamodule=datamodule)

    # 测试可视化一下结果
    task.load_state_dict(torch.load("./logs/torcgeo_rit18/epoch=4-step=35-v1.ckpt")['state_dict'])
    val_dataset = RIT18(split="val")
    IDS = 0
    sample = val_dataset[IDS]

    image, mask = sample['image'], sample['mask']
    pred = task.model(image.unsqueeze(0))
    pred_mask = torch.argmax(pred, dim=1)

    fig = view_rit18(sample, pred_mask.squeeze(0))
    plt.savefig(f"./assets/rit18_sample_{IDS}.png")