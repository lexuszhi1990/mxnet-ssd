# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.You can use the code to train/evaluate/test for object detection task.

### project overview

```
deploy
├── c++
│   ├── main.cc
│   ├── readme.md
│   ├── CMakeLists.txt
│   └── cmake
├── python
│   ├── main.py
│   └── readme.md
├── models
│   └── <model_list>
└── demo_images
    └── <demo_images>
```

### init

```
data dir
/mnt/data/:dataset_name/
/mnt/data/:dataset_name/val
/mnt/data/:dataset_name/train
/mnt/data/:dataset_name/test

job dir
# /mnt/jobs/:job_id/config
# /mnt/jobs/:job_id/ckpt/

train logs and metrics
/mnt/training_logs/:job_id/

training model management
/mnt/models

### docker setup

docker run -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-mirror:/app mxnet-ssd-bike:v0.2.1 bash

docker run -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-mirror:/app mxnet-cu90-ssd:v0.1 bash

for 172 server:
docker run --network host -it --rm -v /data/david/models/ssd:/mnt/models -v /data/david/cocoapi:/mnt/datasets/coco -v /data/david/VOCdevkit:/mnt/datasets/voc -v /home/david/mxnet-ssd:/app mxnet-cu90-ssd:v0.1 bash

### training

densenet121(35M):
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3.6 train.py --train-path /mnt/datasets/coco/build/person/train-total.rec --val-path /mnt/datasets/coco/build/person/val-total.rec --network densenet121 --data-shape 480 270 --label-width 480 --lr 0.004 --lr-steps 32,64,96  --end-epoch 128 --num-class 1 --class-names person --prefix /mnt/models/train-v1/ssd --gpus 0,1,2,3 --batch-size 142

CUDA_VISIBLE_DEVICES=6 python3.6 evaluate.py --rec-path /mnt/datasets/coco/build/person/val-total.rec --network densenet121 --data-shape 480 270 --num-class 1 --class-names person --prefix /app/output/exp1/ssd --epoch 28 --batch-size 16 --gpus 0
```

inceptionv3(98M):

```
CUDA_VISIBLE_DEVICES=6 python3.6 evaluate.py --rec-path /mnt/datasets/coco/build/person/val-total.rec --network inceptionv3 --data-shape 480 270 --num-class 1 --class-names person --prefix /mnt/models/ssd-inception-v1/ssd --epoch 28 --batch-size 16 --gpus 0
```

/mnt/models/ssd-inception-v1/ssd-0069.params


### bak

docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-ssd:v0.1 bash

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --num-class 1 --topk 100 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --num-class 1 --topk 100 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 128 --num-class 4 --topk 400 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix ./upload/model/V1/deploy_ssd-densenet-tiny-ebike-detection --epoch 128  --num-class 2 --topk 400 --threshold 0.30

python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --class-names person --images ./data/demo/street.jpg

python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/deploy_ssd-1-1 --epoch 150 --deploy --class-names person --images ./data/demo/street.jpg
```

### daliy log


filter person images( `float(bbox[2]) * float(bbox[3]) < 48*96` ) :

train images: 54368, and anno num: 122683
val images: 2271, and anno num: 5239


train with refine person:

CUDA_VISIBLE_DEVICES=2,3 python3.6 train.py --train-path /mnt/datasets/coco/build/person/train-refined.rec --val-path /mnt/datasets/coco/build/person/val-refined.rec --network inceptionv3 --data-shape 480 270 --label-width 480 --lr 0.04 --lr-steps 20,60,80 --end-epoch 128 --num-class 1 --class-names person --prefix /mnt/models/train-inception-v2/ssd --gpus 0,1 --batch-size 64

CUDA_VISIBLE_DEVICES=2,3 python3.6 train.py --train-path /mnt/datasets/voc/build/person-train.rec --val-path /mnt/datasets/voc/build/person-val.rec --network inceptionv3 --data-shape 480 270 --label-width 480 --lr 0.04 --lr-steps 20,60,80,100 --end-epoch 128 --num-class 1 --class-names person --prefix /mnt/models/train-inception-v2/ssd --gpus 0,1 --batch-size 64


CUDA_VISIBLE_DEVICES=0 python3.6 evaluate.py --rec-path /mnt/datasets/voc/build/person-val.rec --network inceptionv3 --data-shape 480 270 --num-class 1 --class-names person --prefix /mnt/models/ssd-inception-v1/ssd --epoch 121 --batch-size 16 --gpus 0


#### build pascal voc person dataset:

train:
images: 9540, and anno num: 17161
... remaining 17161/17161 labels.
filtering images with no gt-labels. can abort filtering using *true_negative* flag
... remaining 8624/9540 images.
saving list to disk...

validate:
images: 2000, and anno num: 4271
... remaining 4271/4271 labels.
filtering images with no gt-labels. can abort filtering using *true_negative* flag
... remaining 1914/2000 images.

### 2018.7.16

inceptionv3:

|model\dataset|total-person|refined-perspn|VOC2017|
|------------|------------|--------------|-------|
|total(480x270)|0.3511|0.4957|0.556|
|refined(480x270)|0.2652|0.5272|0.477|
|total(360x360)|0.3583|0.7074|0.7662|
|total(360x360)-coorinate|0.4436|0.6975|TBD|


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3.6 train.py --train-path /mnt/datasets/coco/build/person/train-total.rec --val-path /mnt/datasets/coco/build/person/val-total.rec --network inceptionv3 --data-shape 360 360 --label-width 360 --lr 0.04 --lr-steps 80,120 --end-epoch 128 --num-class 1 --class-names person --prefix /mnt/models/train-inception-v4/ssd --gpus 0,1,2,3,4,5,6,7 --batch-size 320 --tensorboard True

CUDA_VISIBLE_DEVICES=7 python3.6 evaluate.py --rec-path /mnt/datasets/coco/build/person/val-total.rec --network inceptionv3 --data-shape 360 360 --num-class 1 --class-names person --prefix /mnt/models/train-inception-v4/ssd --epoch 27 --batch-size 4 --gpus 0

CUDA_VISIBLE_DEVICES=7 python evaluate.py --rec-path /mnt/datasets/coco/build/person/val-total.rec --network inceptionv3 --data-shape 360 360 --num-class 1 --class-names person --prefix /mnt/models/train-inception-v5/ssd --epoch 300 --batch-size 16 --gpus 5

python3.6 demo.py --network inceptionv3 --cpu --class-names person --data-shape 360 --prefix /mnt/models/train-inception-v4/ssd --epoch 256 --class-names person --images ./samples/demo/ebike-three.jpg

python deploy.py --network inceptionv3 --prefix /mnt/models/train-inception-v5/ssd --epoch 380 --num-class 1 --topk 100
