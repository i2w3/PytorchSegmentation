from pathlib import Path
from typing import Union, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

class suffixDataset(Dataset):
    '''扫描对应路径下的文件，要求图像与掩膜放在同级目录下
    '''
    def __init__(self, 
                 datasetPath:Path, 
                 transforms:dict[str:transforms], 
                 suffix:List[str], 
                 Logger:Optional[object] = None,
                 mode:str = "P") -> None:
        super().__init__()
        self.datasetPath = datasetPath
        self.transforms = transforms
        self.suffix = suffix
        self.logger = Logger   
        self.mode = mode 
        self.img, self.mask = self._scanDataset()

    def __getitem__(self,index) -> Union[torch.Tensor, torch.Tensor]:
        img = self.img[index]
        mask = self.mask[index]
        
        img_open = Image.open(img)
        mask_open = Image.open(mask).convert(self.mode)

        if self.transforms is not None:
            img_tensor = self.transforms["img"](img_open)
            mask_tensor = self.transforms["mask"](mask_open)
        mask_tensor = torch.squeeze(mask_tensor) # (1, 256, 256) -> (256, 256)
        return img_tensor, mask_tensor
    
    def __len__(self):
        return len(self.img)
    
    def _scanDataset(self) -> Union[List[Path], List[Path]]:
        '''将maskSuffix扫描到的mask替换为imgSuffix即为img
        '''
        imgSuffix, maskSuffix = self.suffix
        mask = [file for file in self.datasetPath.rglob("*" + maskSuffix)]
        img = [file.with_name(file.name.replace(maskSuffix, imgSuffix)) for file in mask]

        # img = [file for file in self.datasetPath.rglob("*" + imgSuffix) if maskSuffix not in file.name]
        # mask = [file.with_name(file.name.replace(imgSuffix, maskSuffix)) for file in img]
        if self.logger is not None:
            self.logger("info", f"DataSet Configuration\n{vars(self)}")
        return img, mask
    
