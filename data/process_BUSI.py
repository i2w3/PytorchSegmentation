# Breast Ultrasound Images Dataset
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

datasetPath = Path("./data/Dataset_BUSI_with_GT")
processedPath = datasetPath / "processed"

if processedPath.exists() and processedPath.is_dir():
    shutil.rmtree(processedPath)
Path.mkdir(processedPath)

imgSuffix = ".png"
maskSuffix = "_mask.png"
newMaskSuffix = "_mask_*.png"

classes = {"normal":0, "benign":1, "malignant": 2}
palette = [  0, 0, 0,
           255, 0, 0,
           0, 255, 0]

for key, value in classes.items():
    files = list((datasetPath / key).rglob("*" + maskSuffix))
    for file in files:
        mask = np.array(Image.open(file).convert("L"))
        rglob = file.with_name(file.name.replace(maskSuffix, newMaskSuffix)).name
        extra_files = list((datasetPath / key).rglob(rglob))
        if extra_files != []:
            print(extra_files)
            for extra_file in extra_files:
                mask += np.array(Image.open(extra_file).convert("L"))
            print(np.unique(mask))
        # mask[mask==255] = value
        # mask[mask==254] = value
        mask[mask > 0] = value
        mask = Image.fromarray(mask).convert("P")
        mask.putpalette(palette)
        mask.save(processedPath / file.name)
        
        shutil.copy(file.with_name(file.name.replace(maskSuffix, imgSuffix)), processedPath)
