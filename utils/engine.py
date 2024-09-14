import math
import pickle
import argparse

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from .metrics import ConfusionMatrix
from .utils import MaskOverflow
    
class Engine():
    def __init__(self, 
                 model:nn.Module, 
                 args:argparse.Namespace,
                 logger:object,
                 keys_list:list = ["train_loss", "train_precision", "train_iou", "train_dice",
                                   "valid_loss", "valid_precision", "valid_iou", "valid_dice"]) -> None:
        self.model = model
        self.args = args
        self.logger = logger
        self.LoggingPath = self.logger.log_path.parent
        self.loss = nn.CrossEntropyLoss(weight = torch.from_numpy(np.ones([args.num_classes], np.float32)).cuda())
        self.optimizer = torch.optim.Adam(model.parameters(),lr = args.lr, weight_decay = args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        self.sroce: dict[str, list] = {key:[] for key in keys_list}

        self.cm = ConfusionMatrix(args.num_classes)
        self.best_sroce = 0

        self.log("info", f"loss {self.loss.__class__.__name__}, optimizer {self.optimizer.__class__.__name__}, scheduler {self.scheduler.__class__.__name__}")
        self.log("info", f"args {args}")

    def log(self, *args: str) -> None:
        self.logger(*args)

    def updateSroce(self, sroce:float, ddp:bool = False) -> None:
        state:dict = {'model_state': self.model.module.state_dict() if ddp else self.model.state_dict(),
                      #'optimizer': self.optimizer
                      }
        if sroce > self.best_sroce:
            isBest = True
            self.best_sroce = sroce
            self.log("info", f"saving best model")
        else:
            isBest = False
        torch.save(state, self.LoggingPath / ('best.pth' if isBest else 'last.pth'))

    def saveSroce(self, enableSaving:bool = False, **kwargs) -> None:
        for key, value in kwargs.items():
            self.sroce[key].append(value)
        self.log("info", kwargs)
        if enableSaving:
            with open(self.LoggingPath / "sroce.pkl", "wb") as f:
                pickle.dump(self.sroce, f, pickle.HIGHEST_PROTOCOL)
    
    def plotSroce(self, row_plots:int = 2, figSize:tuple = (4,4)) -> None:
        width, height = figSize
        num_plots = len(self.sroce)
        col_plots = math.ceil(num_plots / row_plots)  # 每行显示的子图数量
        fig, axes = plt.subplots(row_plots, col_plots, figsize=(col_plots * width, row_plots * height))

        axes = axes.flatten() # 将axes展平成一维数组

        # 遍历字典并绘图
        for idx, (key, values) in enumerate(self.sroce.items()):
            ax = axes[idx]
            ax.plot(values)
            ax.set_title(key)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # 设置x轴刻度为整数
        
        # 隐藏多余的子图
        for i in range(num_plots, row_plots * col_plots):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig(self.LoggingPath / "sroce.png")

    def train_epoch(self, trainDataLoader):
        running_loss = 0.0
        self.cm.reset()

        self.model.train()
        for x, y in tqdm(trainDataLoader, desc="train"):
            self.optimizer.zero_grad()
            x, y = x.to('cuda'), y.to('cuda')
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                running_loss += loss.item()
                self.cm.update(y_pred, y)

        train_loss = running_loss / len(trainDataLoader.dataset)

        metrics = self.cm.calculate_metrics()
        self.saveSroce(train_loss=round(train_loss, 3), 
                       train_precision=round(metrics.mprecision, 3), 
                       train_iou=round(metrics.miou, 3), 
                       train_dice=round(metrics.mdice, 3))
    
    @torch.inference_mode()
    def valid_epoch(self, validDataLoader) -> None:
        running_loss = 0 
        self.cm.reset()
        self.model.eval()
        self.scheduler.step()

        for x, y in tqdm(validDataLoader, desc="valid"):
            x, y = x.to('cuda'), y.to('cuda')
            if (y.min() < 0) or (y.max() >= self.args.num_classes):
                raise MaskOverflow
            y_pred = self.model(x)
            loss = self.loss(y_pred, y)
            running_loss += loss.item()
            self.cm.update(y_pred, y)

        valid_loss = running_loss / len(validDataLoader.dataset)
        metrics = self.cm.calculate_metrics()
        
        self.saveSroce(True, 
                       valid_loss=round(valid_loss, 3), 
                       valid_precision=round(metrics.mprecision, 3), 
                       valid_iou=round(metrics.miou, 3),
                       valid_dice=round(metrics.mdice, 3))
        self.updateSroce(metrics.miou)
