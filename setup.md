# pedestrian detection

### local dev

python3.6 -m venv ssd-env
source ssd-env/bin/activate
pip install --upgrade pip

pip install -i https://mirrors.aliyun.com/pypi/simple/ mxnet-cu80 opencv-python matplotlib pyyaml

### start docker env

start cpu docker image:

`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-ssd:v0.1 bash`

start gpu docker image:

`docker run --network host -it --rm -v /home/fulingzhi/workspace/mxnet-ssd-pedestrian:/app -v /mnt/gf_mnt/datasets:/mnt/datasets -v /mnt/gf_mnt/jobs:/mnt/jobs  mxnet-cu90-ssd:v0.1 bash`


### train




### evaluate

cpu:
`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job2/ssd-1-1 --epoch 70 --cpu --voc07`

gpu:
`python3.6 evaluate.py --rec-path /mnt/datasets/coco_person/val.rec --list-path /mnt/datasets/coco_person/val.lst --network densenet-tiny --batch-size 1 --num-class 1 --class-names person --prefix /mnt/jobs/job2/ssd-1-1 --epoch 70 --gpu 0`


### deploy

`python3.6 deploy.py --network densenet-tiny --prefix ./upload/model/V1/deploy_ssd-densenet-tiny-ebike-detection --epoch 128  --num-class 2 --topk 400 --threshold 0.30`


### demo

`python3.6 demo.py --network densenet-tiny --cpu --class-names person --data-shape 300 --prefix /mnt/jobs/job2/ssd-1-1 --epoch 150 --class-names person --images ./data/demo/street.jpg`
