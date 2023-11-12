from utils import get_model
from mmdet.apis import inference_detector, show_result_pyplot
import glob
import pandas as pd 
import os
from tqdm import tqdm
import mmcv
from mmcv import imread
import numpy as np
weights_path = '/data/operati123/yolo/model/epoch_45.pth'
config_file_path='./yolo_config_test.py'
model=get_model(config_file_path,weights_path)

test_directory = '/local_datasets/Test2'
test_file = glob.glob(os.path.join(test_directory, '*.jpg'))

results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}
score_threshold = 0.4
for index, img_path in tqdm(enumerate(test_file), total = len(test_file)):

    file_name = img_path.split("/")[-1].split(".")[0]+".json"

    img = mmcv.imread(img_path)
    predictions = inference_detector(model, img)
    boxes, scores, labels = (list(), list(), list())

    for k, cls_result in enumerate(predictions):
        # print("cls_result", cls_result)
        if cls_result.size != 0:
            if len(labels)==0:
                boxes = np.array(cls_result[:, :4])
                scores = np.array(cls_result[:, 4])
                labels = np.array([k+1]*len(cls_result[:, 4]))
            else:
                boxes = np.concatenate((boxes, np.array(cls_result[:, :4])))
                scores = np.concatenate((scores, np.array(cls_result[:, 4])))
                labels = np.concatenate((labels, [k+1]*len(cls_result[:, 4])))

    if len(labels) != 0:
        indexes = np.where(scores > score_threshold)
        # print(indexes)
        boxes = boxes[indexes]
        scores = scores[indexes]
        labels = labels[indexes]

        for label, score, bbox in zip(labels, scores, boxes):
            x_min, y_min, x_max, y_max = bbox.astype(np.int64)

            results['file_name'].append(file_name)
            results['class_id'].append(label)
            results['confidence'].append(score)
            results['point1_x'].append(x_min)
            results['point1_y'].append(y_min)
            results['point2_x'].append(x_max)
            results['point2_y'].append(y_min)
            results['point3_x'].append(x_max)
            results['point3_y'].append(y_max)
            results['point4_x'].append(x_min)
            results['point4_y'].append(y_max)

submission = pd.DataFrame(results)
save_path = '/data/operati123/yolo/submission.csv'
submission.to_csv(save_path, index=False)
