from mmdet.apis import init_detector, inference_detector
import mmcv
import os
import shutil

from glob import glob


path = "/local_datasets/aircraftAndBird" # train.txt와 val.txt, test.txt를 저장할 path

img_path= "/local_datasets/aircraftAndBird/training2/images" # 이미지가 존재하는 디렉토리 경로

train_img_list = glob(f'{img_path}/*') # 이미지 파일들의 경로 주소들을 읽어온 후 리스트로 저장한다.


from sklearn.model_selection import train_test_split

train_img_list, val_img_list = train_test_split(train_img_list, test_size=0.2, random_state=2000)
train_img_list, test_img_list = train_test_split(train_img_list, test_size=0.1, random_state=2000)

with open(f'{path}/train.txt','w') as f: # train을 위한 self.ann_file 만들기
        f.write('\n'.join(train_img_list) + '\n')

with open(f'{path}/val.txt','w') as f: # val을 위한 self.ann_file 만들기
        f.write('\n'.join(val_img_list) + '\n')

with open(f'{path}/test.txt','w') as f: # test을 위한 self.ann_file 만들기
        f.write('\n'.join(test_img_list) + '\n')

import json

def getInformationsFromJson(filePath):
  with open(filePath) as file:
    bbox_names = []
    bboxes = []

    oneJsonFile = json.load(file)

    pure_image_file_name = oneJsonFile["images"][0]["id"]
    width = oneJsonFile["images"][0]["width"]
    height = oneJsonFile["images"][0]["height"]

    for annotation in oneJsonFile["annotations"]:
      category_id= int(annotation["category_id"])
      if (category_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
        category_id = 0 # 그냥 조류:0으로 세팅
      xmin = int(annotation["bbox"][0])
      ymin = int(annotation["bbox"][1])
      xDistance = int(annotation["bbox"][2])
      yDistance = int(annotation["bbox"][3])

      bboxes.append([xmin, ymin, xmin + xDistance, ymin + yDistance])
      bbox_names.append(category_id)

    return pure_image_file_name, bbox_names, bboxes

import copy
import os.path as osp
import mmcv
import numpy as np
import cv2

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class AircraftDataset(CustomDataset):
  CLASSES = ("Bird", "Airplane", "Helicopter", "FighterPlane", "Paragliding", "Drone")

  # annotation에 대한 모든 파일명을 가지고 있는 텍스트 파일을 __init__(self, ann_file)로 입력 받고,
  # 이 self.ann_file이 load_annotations()의 인자로 입력
  def load_annotations(self, ann_file): # ann_file은 train.txt이다.
    cat2label = {
        0: "Bird", 223:"Airplane", 224:"Helicopter", 225:"FighterPlane", 226:"Paragliding", 227:"Drone"
    }
    image_file_address_path_list = mmcv.list_from_file(self.ann_file)
    # 포맷 중립 데이터를 담을 list 객체
    data_infos = []

    for image_file_address_path in image_file_address_path_list:
      # 원본 이미지의 너비, 높이를 image를 직접 로드하여 구함.
      image = cv2.imread(image_file_address_path)
      height, width = image.shape[:2]

      # 개별 annotation json 파일이 있는 서브 디렉토리의 prefix 변환.
      annotation_json_file_address_path = image_file_address_path.replace('images', 'labels')
      annotation_json_file_address_path = annotation_json_file_address_path.replace('jpg', 'json')
      label_prefix = self.img_prefix.replace('images', 'labels')

      # getInformationsFromJson() 를 이용하여 개별 json 파일에 있는 이미지의 모든 bbox 정보를 list 객체로 생성.
      pure_image_file_name, bbox_names, bboxes = getInformationsFromJson(annotation_json_file_address_path)

      # 개별 annotation json 파일을 1개 line 씩 읽어서 list 로드.
      anno_json_file = osp.join(label_prefix, str(pure_image_file_name) + '.json')
     # 이미지 파일은 있으나 json annotation 파일이 없는 경우는 제외.
      if not osp.exists(anno_json_file):
          continue

      # 개별 image의 annotation 정보 저장용 Dict 생성. key값 filename에는 image의 파일명만 들어감(디렉토리는 제외)
      # 영상에는 data_info = {'filename': filename 으로 되어 있으나 filename은 image 파일명만 들어가는게 맞음.
      data_info = {'filename': str(pure_image_file_name) + '.jpg',
                  'width': width, 'height': height}

      gt_bboxes = []
      gt_labels = []
      gt_bboxes_ignore = []
      gt_labels_ignore = []

      # bbox별 Object들의 class name을 class id로 매핑. class id는 tuple(list)형의 CLASSES의 index값에 따라 설정
      for bbox_name, bbox in zip(bbox_names, bboxes):
        # 만약 bbox_name이 클래스명에 해당 되면, gt_bboxes와 gt_labels에 추가, 그렇지 않으면 gt_bboxes_ignore, gt_labels_ignore에 추가
        # bbox_name이 CLASSES중에 반드시 하나 있어야 함. 안 그러면 FILTERING 되므로 주의 할것.
        if bbox_name in cat2label:
            gt_bboxes.append(bbox)
            # gt_labels에는 class id를 입력
            if (bbox_name == 0):
              gt_labels.append(bbox_name)
            else:
              gt_labels.append(bbox_name-222)
        else:
            gt_bboxes_ignore.append(bbox)
            gt_labels_ignore.append(-1)

      # 개별 image별 annotation 정보를 가지는 Dict 생성. 해당 Dict의 value값을 np.array형태로 bbox의 좌표와 label값으로 생성.
      data_anno = {
        'bboxes': np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
        'labels': np.array(gt_labels, dtype=np.long),
        'bboxes_ignore': np.array(gt_bboxes_ignore, dtype=np.float32).reshape(-1, 4),
        'labels_ignore': np.array(gt_labels_ignore, dtype=np.long)
      }

      # image에 대한 메타 정보를 가지는 data_info Dict에 'ann' key값으로 data_anno를 value로 저장.
      data_info.update(ann=data_anno)
      # 전체 annotation 파일들에 대한 정보를 가지는 data_infos에 data_info Dict를 추가
      data_infos.append(data_info)
      #print(data_info)

    return data_infos

# config 파일은 faster rcnn resnet 50 backbone 사용.
# 학습으로 생성된 모델을 Google Drive에 저장

# edited here
config_file = "./mmdetection/configs/yolof/yolof_r50_c5_8x8_1x_coco.py"
checkpoint_file = './mmdetection/checkpoints/yolof_r50_c5_8x8_1x_coco_20210425_024427-8e864411.pth'
# edited here

# Google Drive 밑에 Directory 생성. 이미 생성 되어 있을 시 오류 발생.
#if os.path.exists('./aircraft_work_dir'): shutil.rmtree("./aircraft_work_dir")
#os.mkdir("./aircraft_work_dir")

from mmcv import Config

cfg = Config.fromfile(config_file)
from mmdet.apis import set_random_seed


data_root = '/local_datasets/aircraftAndBird/'
# dataset에 대한 환경 파라미터 수정.
# dataset에 대한 환경 파라미터 수정.
cfg.dataset_type = 'AircraftDataset'
cfg.data_root = data_root

# train, val, test dataset에 대한 type, data_root, ann_file, img_prefix 환경 파라미터 수정.
cfg.data.train.type = 'AircraftDataset'
cfg.data.train.data_root='/local_datasets/aircraftAndBird/'
cfg.data.train.ann_file = data_root + 'train.txt'
cfg.data.train.img_prefix = data_root + 'training2/images'


cfg.data.val.type = 'AircraftDataset'
cfg.data.val.data_root='/local_datasets/aircraftAndBird/'
cfg.data.val.ann_file = data_root + 'val.txt'
cfg.data.val.img_prefix = data_root +'training2/images'

cfg.data.test.type = 'AircraftDataset'
cfg.data.test.data_root='/local_datasets/aircraftAndBird/'
cfg.data.test.ann_file = data_root + 'test.txt'
cfg.data.test.img_prefix = data_root + 'training2/images'

# class의 갯수 수정.
cfg.model.bbox_head.num_classes = 6
# pretrained 모델
cfg.load_from = checkpoint_file

# 학습 weight 파일로 로그를 저장하기 위한 디렉토리로 구글 Drive 설정.
cfg.work_dir = './final'

# 학습율 변경 환경 파라미터 설정.
cfg.optimizer.lr = 0.02 / 8
# cfg.lr_config.warmup_iters=1000
# 1번 마다 한번씩 로그를 작성
cfg.log_config.interval = 1

cfg.runner.max_epochs = 15 # edited here

# 평가 metric 설정.
cfg.evaluation.metric = 'mAP'
# 평가 metric 수행할 epoch interval 설정.
cfg.evaluation.interval = 1
# 학습 iteration시마다 모델을 저장할 epoch interval 설정.
cfg.checkpoint_config.interval = 1

# 학습 시 Batch size 설정(단일 GPU 별 Batch size로 설정됨) (gpu가 2개이면 batch size가 8이 된다.)
cfg.data.samples_per_gpu = 16 # default 4

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
# 두번 config를 로드하면 lr_config의 policy가 사라지는 오류로 인하여 설정.
cfg.lr_config.policy='step'

# ConfigDict' object has no attribute 'device 오류 발생시 반드시 설정 필요. https://github.com/open-mmlab/mmdetection/issues/7901
cfg.device='cuda'

cfg.dump('yolo_config_test.py') # 수정된 config 파일 저장
# We can initialize the logger for training and have a look
# at the final config used for training
# print(f'Config:\n{cfg.pretty_text}')

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
print(cfg.data.train)
# train용 Dataset 생성.
datasets = [build_dataset(cfg.data.train)]
print(datasets)
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES

'''
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회
train_detector(model, datasets, cfg, distributed=False, validate=True)
'''
model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
model.CLASSES = datasets[0].CLASSES


mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
# epochs는 config의 runner 파라미터로 지정됨. 기본 12회
# 3시간 소모
train_detector(model, datasets, cfg, distributed=False, validate=True)
