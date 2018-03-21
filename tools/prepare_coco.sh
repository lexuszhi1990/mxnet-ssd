#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3.6 $DIR/prepare_dataset.py --dataset coco --set train2017 --target /mnt/gf_mnt/datasets/coco_bike_person/train-bike-total.lst  --root /mnt/gf_mnt/datasets/cocoapi
python3.6 $DIR/prepare_dataset.py --dataset coco --set val2017 --target /mnt/gf_mnt/datasets/coco_bike_person/val-bike-total.lst --shuffle False --root /mnt/gf_mnt/datasets/cocoapi
