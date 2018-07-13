#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3.6 $DIR/prepare_dataset.py --root /mnt/datasets/voc --dataset pascal --year 2012 --set new_trainval --target /mnt/datasets/voc/build/person-train.lst
python3.6 $DIR/prepare_dataset.py --root /mnt/datasets/voc --dataset pascal --year 2012 --set new_val --target /mnt/datasets/voc/build/person-val.lst --shuffle False
