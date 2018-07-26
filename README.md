# Pedestrian Detection based on SSD

SSD is an unified framework for object detection with a single network.You can use the code to train/evaluate/test for object detection task.

### what's new

- support multi-machines distributed training via ssh/k8s
- support arbitrary shape for input
- support CoordinateConv

### results

pedestrian detection(inceptionv3):

|model\dataset|total-person|refined-perspn|VOC2017|
|------------|------------|--------------|-------|
|total(480x270)|0.3511|0.4957|0.556|
|refined(480x270)|0.2652|0.5272|0.477|
|total(360x360)|0.3583|0.7074|0.7662|
|total(360x360)-coorinate|0.4436|0.6975|0.75|


### Demo results

![image](./data/demo/street_demo.png "person")

### Usage

demo:

`python demo.py --network inceptionv3 --class-names person --data-shape 360 --prefix your-model-path --epoch your-epoch-num --class-names person --images ./data/demo/street.jpg`

training:

`python train.py --train-path /mnt/datasets/coco/build/person/train-total.rec --val-path /mnt/datasets/coco/build/person/val-total.rec --network densenet121 --data-shape 360 360 --label-width 360 --lr 0.004 --lr-steps 32,64,96 --end-epoch 128 --num-class 1 --class-names person`

evaluate:

`python evaluate.py --rec-path /mnt/datasets/coco/build/person/val-total.rec --network inceptionv3 --data-shape 360 360 --num-class 1 --class-names person --prefix /mnt/models/ssd-inception-v1/ssd --epoch 28 --batch-size 16 --gpus 0`

