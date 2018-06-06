# SSD: Single Shot MultiBox Object Detector

SSD is an unified framework for object detection with a single network.

You can use the code to train/evaluate/test for object detection task.

### project overview

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

### init

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

### docker env

docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-ssd:v0.1 bash

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --num-class 1 --topk 100 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --num-class 1 --topk 100 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix /mnt/jobs/job2/ssd-1-1 --epoch 128 --num-class 4 --topk 400 --threshold 0.30

python3.6 deploy.py --network densenet-tiny --prefix ./upload/model/V1/deploy_ssd-densenet-tiny-ebike-detection --epoch 128  --num-class 2 --topk 400 --threshold 0.30

python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --class-names person --images ./data/demo/street.jpg

python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/deploy_ssd-1-1 --epoch 150 --deploy --class-names person --images ./data/demo/street.jpg

```
arg_shapes, out_shapes, aux_shapes=out.infer_shape(**{"data": (1, 3, 300, 300)})
```
