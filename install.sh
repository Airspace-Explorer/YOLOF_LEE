#!/bin/sh

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13/index.html

git clone --branch 2.x https://github.com/open-mmlab/mmdetection.git
cd mmdetection
python setup.py install
pip uninstall yapf -y
pip install yapf==0.40.1

mkdir checkpoints
!wget -O ./mmdetection/checkpoints/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_mstrain-608_273e_coco/yolov3_d53_mstrain-608_273e_coco_20210518_115020-a2c3acb8.pth
pip install scipy matplotlib pycocotools scikit-learn
pip install numpy==1.20.0

