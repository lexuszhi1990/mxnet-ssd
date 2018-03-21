#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
python3.6 $DIR/prepare_dataset.py --dataset bike --set train --target /home/fulingzhi/workspace/mxnet-ssd-bike/data/bike/train-bike.lst  --root /home/fulingzhi/workspace/mxnet-ssd-bike/data/bike --class-names bike,ebike,irrelevant
python3.6 $DIR/prepare_dataset.py --dataset bike --set train --target /home/fulingzhi/workspace/mxnet-ssd-bike/data/bike/val-bike.lst  --root /home/fulingzhi/workspace/mxnet-ssd-bike/data/bike --class-names bike,ebike,irrelevant
