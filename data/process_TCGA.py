import shutil
from pathlib import Path

import numpy as np
from PIL import Image

image_path = Path("./data/kaggle_3m") 
image_list = image_path.rglob("*_mask.tif")
save_path = Path("./data/kaggle_3m_processed")
Path.mkdir(save_path)

palette = [  0, 0, 0,
           255, 0, 0]

for image in image_list:
    image_data = np.array(Image.open(image))
    image_data[image_data==255] = 1
    if np.all(np.unique(image_data) == 0):
        continue
    image_data = Image.fromarray(image_data).convert("P")
    image_data.putpalette(palette)
    image_data.save(save_path / image.with_suffix(".png").name)

    image_raw_path = Path(str(image).replace("_mask.tif", ".tif"))
    image_raw = Image.open(image_raw_path)
    image_raw.save(save_path / image_raw_path.with_suffix(".png").name)
