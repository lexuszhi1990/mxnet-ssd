#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3.6 $DIR/prepare_dataset.py --dataset bike --set train --target /mnt/data/bike/train-bike.lst  --root /mnt/data/bike --class-names bike,ebike,irrelevant
python3.6 $DIR/prepare_dataset.py --dataset bike --set train --target /mnt/data/bike/val-bike.lst  --root /mnt/data/bike --class-names bike,ebike,irrelevant
