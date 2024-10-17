import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms
import numpy as np
from PIL import Image


from utils import *
from models import UNet, init_weights

PALETTE = [  0,   0,   0,
           128, 128, 128,
           255, 255, 255]

def get_args_parser():
    parser = argparse.ArgumentParser('todo', add_help=False)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--test_number', default=10, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    transforms_dict = {"img": transforms.Compose([transforms.ToImage(),
                                                  transforms.ToDtype(torch.uint8, scale=True), 
                                                  transforms.Resize((512,512)),
                                                  transforms.ToDtype(torch.float32, scale=True),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                  ]),
                       "mask": transforms.Compose([transforms.ToImage(),
                                                   transforms.ToDtype(torch.uint8), 
                                                   transforms.Resize((512,512), interpolation=0),
                                                   transforms.ToDtype(torch.long),
                                                   ]),}
    
    model = UNet(num_classes=args.num_classes)
    model = init_weights(model)
    engine = Engine(model.to('cuda'), args, Logger("Trainer"))

    dataset = suffixDataset(Path(r"./data/Dataset_BUSI_with_GT/processed"), 
                            transforms_dict, 
                            [".png", "_mask.png"], 
                            Logger("DateSet", engine.logger.log_path))

    train_size = int(0.8 * len(dataset))
    train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

    dl_train = DataLoader(train_dataset, batch_size = args.batch_size, num_workers = 8, shuffle = True)
    dl_valid = DataLoader(valid_dataset, batch_size = args.batch_size, num_workers = 8, shuffle = False)

    for epoch in range(args.epochs):
        engine.log("info", f"{'-'*10}EPOCH {epoch+1}{'-'*10}")
        engine.train_epoch(dl_train)
        engine.valid_epoch(dl_valid)
    engine.plotSroce()

    engine.model = engine.model.eval()
    indices = random.sample(range(len(valid_dataset)), 10)
    for i, indices in enumerate(indices):
        image_data, mask_data = valid_dataset[indices]
        image_data = torch.unsqueeze(image_data, dim=0).cuda()
        with torch.inference_mode():
            
            pred = engine.model(image_data)
            pred = torch.argmax(pred, dim=1)
            pred = torch.squeeze(pred)

            pred = (pred.cpu().numpy()).astype(np.uint8)
            mask = PaletteArray(pred, PALETTE, args.num_classes)
            image = Image.fromarray(mask)
            image.save(engine.LoggingPath / f"test_pred_{i+1}.png")

            mask_data = (mask_data.cpu().numpy()).astype(np.uint8)
            mask = PaletteArray(mask_data, PALETTE, args.num_classes)
            image = Image.fromarray(mask)
            image.save(engine.LoggingPath / f"test_mask_{i+1}.png")
    