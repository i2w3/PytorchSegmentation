import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2 as transforms

from utils import *
from models import UNet, init_weights

def get_args_parser():
    parser = argparse.ArgumentParser('todo', add_help=False)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

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
    
    model = UNet(num_classes=args.num_classes)
    model = init_weights(model)
    engine = Engine(model.to('cuda'), args, Logger("Trainer"))

    dataset = suffixDataset(Path(r"./data/kaggle_3m"), 
                            transforms_dict, 
                            [".tif", "_mask.png"], 
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
    