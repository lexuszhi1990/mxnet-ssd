# pedestrian detection

### local dev

python3.6 -m venv ssd-env
source ssd-env/bin/activate
pip install --upgrade pip

pip install -i https://mirrors.aliyun.com/pypi/simple/ mxnet-cu80 opencv-python matplotlib pyyaml

### setup glusterfs

sudo mount -t glusterfs node1:/my_pg /mnt/gf_mnt/my_pg
sudo mount -t glusterfs node1:/jobs /mnt/gf_mnt/jobs
sudo mount -t glusterfs node1:/datasets /mnt/gf_mnt/datasets
sudo mount -t glusterfs node1:/models /mnt/gf_mnt/models
sudo mount -t glusterfs node1:/scripts /mnt/gf_mnt/scripts

### start docker env

#### on 177 server:
start cpu docker image:
`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs mxnet-ssd:v0.1 bash`

start gpu docker image:
`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-cu90-ssd:v0.1 bash`

`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-mirror:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-cu90-ssd:v0.1 bash`

#### on 172 server:
`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/gfs/fl/datasets -v /mnt/gf_mnt/jobs:/gfs/fl/jobs  mxnet-cu90-ssd:v0.1 bash`

### train

python3.6 train.py --train-path /mnt/datasets/coco_person/train.rec --val-path /mnt/datasets/coco_person/val.rec --prefix /mnt/jobs/vgg16_reduced-v1/ssd --batch-size 8 --data-shape 512 --label-width 512 --lr 0.001 --network vgg16_reduced --tensorboard True --num-class 1 --class-names person --gpu 0

distributed train:
```
python3.6 train.py --train-path /gfs/fl/datasets/coco_person/train.rec --val-path /gfs/fl/datasets/coco_person/val.rec --prefix /gfs/fl/jobs/vgg16_reduced-v1/ssd --batch-size 16 --data-shape 512 --label-width 512 --lr 0.001 --network vgg16_reduced --tensorboard True --num-class 1 --class-names person --gpu 0 --kv-store=dist_device_sync
```

arbitrary shape

`python3.6 train.py --train-path /mnt/datasets/coco_person/train.rec --val-path /mnt/datasets/coco_person/val.rec --prefix /mnt/jobs/densenet-tiny-v1/ssd --batch-size 8 --data-shape 430 270 --label-width 430 --lr 0.001 --network densenet-tiny --tensorboard True --num-class 1 --class-names person --gpu 0`

### evaluate

cpu:
`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job2/ssd-1-1 --epoch 70 --cpu --voc07`

gpu:
`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job2/ssd --epoch 70 --gpu 0`

`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job1/ssd- --epoch 231 --gpu 0`

`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job1/ssd--1- --epoch 128 --gpu 0`

`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network vgg16_reduced --data-shape 512 --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/vgg16_reduced-v1/ssd --epoch 20 --gpu 0 --voc07 True`


### deploy

`python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job1/ssd- --epoch 231  --num-class 1 --topk 400 --threshold 0.30`

`python3.6 deploy.py --network vgg16_reduced --data-shape 512 --prefix /mnt/jobs/vgg16_reduced-v1/ssd  --num-class 1 --topk 100 --threshold 0.30 --epoch 20`


### demo

`python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --class-names person --images ./data/demo/street.jpg`


##  dev logs


### 2018.5.16

version 0.1:
```
python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job1/ssd- --epoch 231 --gpu 0

person: 0.3529423180154818
mAP: 0.3529423180154818
```

arbitrary shape:

```
python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --data-shape 430 270 --prefix /mnt/jobs/densenet-tiny-v1/ssd --epoch 61 --gpu 0

person: 0.3489304573703142
mAP: 0.3489304573703142
```

```
python3.6 train.py --train-path /mnt/datasets/coco_person/train.rec --val-path /mnt/datasets/coco_person/val.rec --prefix /mnt/jobs/densenet-tiny-v2/ssd --batch-size 8 --data-shape 430 270 --label-width 430 --lr 0.001 --network densenet-tiny --tensorboard True --num-class 1 --class-names person --gpu 0 --epoch 256
```
