import torch
from dataclasses import dataclass

@dataclass
class Metrics:
    mprecision:float
    mrecall:float
    miou:float
    mdice:float


class ConfusionMatrix:
    def __init__(self, num_classes:int):
        self.num_classes = num_classes
        self.epsilon = 1e-10
        self.reset()

    def reset(self):
        # 初始化TP, TN, FP, FN为零张量
        self.tp = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)

    def update(self, outputs:torch.Tensor, targets:torch.Tensor):
        """
        outputs: 模型输出的预测，形状为 (B, C, H, W) 或 (B, C)
        targets: 实际标签，形状为 (B, H, W) 或 (B)
        """
        if outputs.shape != targets.shape:
            # 输出是 logits
            outputs = torch.argmax(outputs, dim=1).long()

        for cls in range(self.num_classes):
            tp = ((outputs == cls) & (targets == cls)).sum().item()
            tn = ((outputs != cls) & (targets != cls)).sum().item()
            fp = ((outputs == cls) & (targets != cls)).sum().item()
            fn = ((outputs != cls) & (targets == cls)).sum().item()

            self.tp[cls] += tp
            self.tn[cls] += tn
            self.fp[cls] += fp
            self.fn[cls] += fn

    def calculate_metrics(self):
        precision = (self.tp) / (self.tp + self.fp + self.epsilon)
        recall = (self.tp) / (self.tp + self.fn + self.epsilon)
        iou = self.tp / (self.tp + self.fp + self.fn + self.epsilon)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + self.epsilon)
        
        mprecision = precision.mean().item()
        mrecall = recall.mean().item()
        miou = iou.mean().item()
        mdice = dice.mean().item()
        return Metrics(mprecision, mrecall, miou, mdice)
