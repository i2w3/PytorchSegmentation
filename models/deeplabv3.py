import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet101,deeplabv3_resnet50

def deeplabv3(num_classes=2):
    model = deeplabv3_resnet50(weights='DEFAULT')
    model.classifier[4] = nn.Conv2d(256, num_classes, 1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, 1)
    return model