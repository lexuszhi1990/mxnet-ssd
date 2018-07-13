#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3.6 $DIR/prepare_dataset.py --dataset coco --set train2017 --target /mnt/datasets/coco/build/person/train-refined.lst --root /mnt/datasets/coco
python3.6 $DIR/prepare_dataset.py --dataset coco --set val2017 --target /mnt/datasets/coco/build/person/val-refined.lst --shuffle False --root /mnt/datasets/coco
