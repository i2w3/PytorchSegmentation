from pathlib import Path

import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import v2 as transforms

from models import UNet
from utils import ConfusionMatrix, PaletteArray

PALETTE = [  0,   0,   0,
           128, 128, 128,
           255, 255, 255]

if __name__ == "__main__":
    transforms_dict = {"img": transforms.Compose([transforms.ToImage(),
                                                  transforms.ToDtype(torch.uint8, scale=True), 
                                                  transforms.Resize((256,256)),
                                                  transforms.ToDtype(torch.float32, scale=True),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                  ]),
                       "mask": transforms.Compose([transforms.ToImage(),
                                                   transforms.ToDtype(torch.uint8), 
                                                   transforms.Resize((256,256), interpolation=0),
                                                   transforms.ToDtype(torch.long),
                                                   ]),}
    model_state_path = Path("./logs/2024-09-14T111741/best.pth")
    model_state_dict = torch.load(model_state_path, weights_only=True)["model_state"]

    image_path = Path("./data/kaggle_3m_processed/TCGA_CS_4942_19970222_14.png")
    mask_path = Path("./data/kaggle_3m_processed/TCGA_CS_4942_19970222_14_mask.png")

    image_data = transforms_dict["img"](Image.open(image_path)) # (3, 256, 256)
    image_data = torch.unsqueeze(image_data, dim=0).cuda() # (1, 3, 256, 256)

    mask_data = transforms_dict["mask"](Image.open(mask_path)) # (1, 256, 256)
    mask_data = torch.unsqueeze(mask_data, dim=0).cuda() # (1, 1, 256, 256)

    cm = ConfusionMatrix(2)
    model = UNet(num_classes=2)
    model.load_state_dict(model_state_dict, strict=True)
    model = model.cuda().eval()

    with torch.inference_mode():
        pred = model(image_data) # (1, 2, 256, 256)
        cm.update(pred, mask_data)
        pred = torch.argmax(pred, dim=1) # (1, 256, 256)

        pred = torch.squeeze(pred) # (256, 256)

        pred = (pred.cpu().numpy()).astype(np.uint8)
        mask = PaletteArray(pred, PALETTE, 2)
        image = Image.fromarray(mask)
        image.save("./demo.png")
        print(cm.calculate_metrics())
        