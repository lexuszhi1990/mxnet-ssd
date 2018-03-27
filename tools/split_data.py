# -*- coding: utf-8 -*-

from pathlib import Path
from random import random

img_dir = './data/demo'
output_dir = './data'

img_path = Path(img_dir)
output_path = Path(output_dir)
train_output_path = output_path / 'train.txt'
val_output_path = output_path / 'val.txt'

train_file=open(train_output_path, 'w+')
val_file=open(val_output_path, 'w+')
for img in img_path.glob('*.jpg'):
    if random() > 0.75:
        train_file.write("{}\n".format(img.stem))
    else:
        val_file.write("{}\n".format(img.stem))

val_file.close()
train_file.close()


