import os
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ["OPENCV_OPENCL_DEVICE"] = "disabled"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
torch.set_float32_matmul_precision('high')

import argparse

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
from torchmetrics import JaccardIndex
import segmentation_models_pytorch as smp

from utils import *

'''
python train_smp.py --arch upernet --encoder_name resnext50_32x4d --encoder_weights ssl --num_classes 9 --enable_compile
'''

def parse_args():
    parser = argparse.ArgumentParser(description="基于 Lightning 的语义分割训练脚本")
    parser.add_argument("--arch", type=str, help="模型架构")
    parser.add_argument("--encoder_name", type=str, default=None, help="编码器名称")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="编码器预训练权重")
    parser.add_argument("--num_classes", default=9, type=int, help="类别数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认为 42")
    parser.add_argument("--export_ckpt", type=str, default=None, help="用于导出 ONNX 格式的检查点路径")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率，默认为 5e-5")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="权重衰减，默认为 0.05")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小，默认为 32")
    parser.add_argument("--enable_compile", action="store_true", help="是否启用 torch.compile 优化")
    return parser.parse_args()


class SegLitModule(L.LightningModule):
    def __init__(self, args:argparse.Namespace):
        super().__init__()
        self.save_hyperparameters(args, ignore=["model"])
        self.model = None
        # 背景类也很重要，不做 ignore_index
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', ignore_index=None)
        self.dice_loss  = smp.losses.DiceLoss(mode='multiclass', ignore_index=None)
        self.miou_all   = JaccardIndex(task="multiclass", num_classes=self.hparams.num_classes, ignore_index=None)
        self.miou_all.reset()

    def configure_model(self):
        if self.model is not None:
            return
        if self.hparams.arch.lower() != "fbffnet":
            self.model = smp.create_model(arch=self.hparams.arch,
                                          encoder_name=self.hparams.encoder_name,
                                          encoder_weights=self.hparams.encoder_weights,
                                          activation=None,
                                          in_channels=3,
                                          classes=self.hparams.num_classes)
            self.model.activation = torch.nn.Softmax(dim=1)
        else:
            self.model = TBFFNet(num_classes=self.hparams.num_classes)
        if self.hparams.enable_compile:
            self.model = torch.compile(self.model)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        images, masks = batch
        masks = masks.long()
        assert masks.ndim == 3
        logits = self(images)
        assert (logits.shape[1] == self.hparams.num_classes) 

        loss = self.focal_loss(logits, masks) + self.dice_loss(logits, masks)

        preds = torch.argmax(logits, dim=1)

        if stage != "train":
            self.miou_all.update(preds, masks)
        self.log(f"{stage}_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def _shared_epoch(self, stage):
        if self.miou_all.update_called:
            miou_all = self.miou_all.compute()
            self.log(f"{stage}_mIoU", miou_all, prog_bar=True, sync_dist=True)
        self.miou_all.reset()
        
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self._shared_epoch("train")

    def on_validation_epoch_end(self):
        self._shared_epoch("val")

    def on_test_epoch_end(self):
        self._shared_epoch("test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        
        warmup_epochs = 10
        
        # Warmup 调度器：线性增长
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.2,  # 从 lr * 0.1 开始
            end_factor=1.0,    # 增长到 lr * 1.0
            total_iters=warmup_epochs
        )
        
        # 主调度器：余弦退火
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs - warmup_epochs,
            eta_min=0.01 * self.hparams.lr
        )
        
        # 串联两个调度器
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]  # 切换 epoch 点
        )
        
        return {"optimizer": optimizer, 
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch",}}

    def on_fit_end(self):
        if self.trainer.is_global_zero:
            print(f"Training complete. Best miou is {self.trainer.checkpoint_callback.best_model_score}.")


if __name__ == "__main__":
    args = parse_args()

    if args.export_ckpt is None:
        L.seed_everything(args.seed)
        train_transform = get_train_transforms()
        valid_transform = get_other_transforms()

        train_dataset = EarthVQA(split="train", augmentation=train_transform)
        valid_dataset = EarthVQA(split="val",   augmentation=valid_transform)
        assert train_dataset.num_classes == args.num_classes
        assert valid_dataset.num_classes == args.num_classes
        print(f"len of train-valid-test is {len(train_dataset)}-{len(valid_dataset)}")

        train_loader = make_loader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = make_loader(valid_dataset, batch_size=args.batch_size, shuffle=False)

        lit_model = SegLitModule(args=args)
        lit_model.configure_model()

        callbacks = [ModelCheckpoint(monitor="val_mIoU", mode='max', save_top_k=1, 
                                     filename='{epoch}-{val_mIoU:.4f}', save_weights_only=True, save_last=True),
                     EarlyStopping(monitor="val_loss", mode="min", patience=30, min_delta=0.001, verbose=False),
                     LearningRateMonitor(logging_interval='epoch')]
        
        logger_path = f"{args.arch}_{args.encoder_name}{'' if args.encoder_weights is None else f'_{args.encoder_weights}'}-{train_dataset.__class__.__name__}"
        logger = [CSVLogger("logs", name=logger_path)]

        trainer = L.Trainer(accelerator="gpu", 
                            devices=[3,6], 
                            # strategy='ddp_find_unused_parameters_false',
                            precision="16-mixed",  
                            max_epochs=200, 
                            sync_batchnorm=True, 
                            num_sanity_val_steps=0, 
                            log_every_n_steps=10, 
                            callbacks=callbacks, 
                            logger=logger,)
        trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    else:
        from pathlib import Path
        import yaml

        chpt_path = Path(args.export_ckpt)
        if chpt_path.is_file():
            pass
        else:
            chpt_list = list(chpt_path.rglob("epoch*.ckpt"))
            if len_chpt_list := len(chpt_list) == 0:
                raise FileNotFoundError(f"No checkpoint found in directory: {chpt_path}")
            elif len_chpt_list > 1:
                print(f"[WARNING] Multiple checkpoints found in directory: {chpt_path}. Using the first one: {chpt_path := chpt_list[0]}")
            else:
                chpt_path = chpt_list[0]
        print(f"Loading checkpoint from: {chpt_path}")
        state_dict = torch.load(chpt_path, map_location="cpu", weights_only=False)

        yaml_path = chpt_path.parent.parent / "hparams.yaml"
        with open(yaml_path, 'r') as f:
            saved_hparams = yaml.safe_load(f)
        print(f"Loaded hyperparameters from {yaml_path}: {saved_hparams}")
        lit_model = SegLitModule(args=saved_hparams)
        saved_onnx = chpt_path.parent / f"{saved_hparams['arch']}_{saved_hparams['encoder_name']}.onnx"
        print(f"Exporting ONNX model to: {saved_onnx}")

        lit_model.configure_model()
        lit_model.load_state_dict(state_dict["state_dict"])
        lit_model = lit_model.eval()

        dummy_input = torch.randn(1, 3, 512, 512)
        dummy_output = lit_model(dummy_input)
        print(f"Dummy output shape: {dummy_output.shape}")

        torch.onnx.export(lit_model.model, (dummy_input,), str(saved_onnx),
                          input_names=['input'],
                          output_names=['output'],
                          verbose=False,
                          dynamo=True,
                          external_data=False
        )

        # if len(list(chpt_path.parent.glob("*.onnx.data"))) == 1:
        #     import onnx
        #     from onnx.external_data_helper import load_external_data_for_model
        #     model_path = chpt_path.with_suffix('.onnx')
        #     model = onnx.load_model(str(model_path), load_external_data=True)
        #     load_external_data_for_model(model, base_dir=str(chpt_path.with_suffix('.onnx.data')))
        #     onnx.save_model(model, str(model_path.with_stem("merged")), save_as_external_data=False)
