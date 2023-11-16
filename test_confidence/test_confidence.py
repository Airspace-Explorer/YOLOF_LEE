from utils import get_model
from mmdet.apis import inference_detector
import glob
import pandas as pd 
import os
from tqdm import tqdm
import mmcv
from mmcv import imread
import numpy as np
from shutil import copyfile
import cv2
import matplotlib
matplotlib.use('Agg')
weights_path = '/data/operati123/yolo/model/epoch_15.pth'
config_file_path='./yolo_config_test.py'
model=get_model(config_file_path,weights_path)
class_names = ['Bird', 'Airplane', 'Helicopter', 'FighterPlane', 'Paragliding', 'Drone']
test_directory = '/local_datasets/Test2'
test_file = glob.glob(os.path.join(test_directory, '*.jpg'))

results = {
    'file_name':[], 'class_id':[], 'confidence':[], 'point1_x':[], 'point1_y':[],
    'point2_x':[], 'point2_y':[], 'point3_x':[], 'point3_y':[], 'point4_x':[], 'point4_y':[]
}
score_threshold = 0.15

save_image_directory = '/data/operati123/yolo/test_image'


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
                labels = np.array([k]*len(cls_result[:, 4]))
            else:
                boxes = np.concatenate((boxes, np.array(cls_result[:, :4])))
                scores = np.concatenate((scores, np.array(cls_result[:, 4])))
                labels = np.concatenate((labels, [k]*len(cls_result[:, 4])))
    
    if len(labels) != 0:
        indexes = np.where(scores > score_threshold)
        # print(indexes)
        boxes = boxes[indexes]
        scores = scores[indexes]
        labels = labels[indexes]

        for label, score, bbox in zip(labels, scores, boxes):
            x_min, y_min, x_max, y_max = bbox.astype(np.int64)
            
            points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], np.int32)
            points = points.reshape((-1, 1, 2))
            img= cv2.polylines(img, [points], isClosed=True, color=(0, 0, 255), thickness=3)
            class_name = class_names[label]  # Assuming label is the class identifier
            text = f"Class: {class_name}, Confidence: {score:.2f}"
            cv2.putText(img, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            save_image_path = os.path.join(save_image_directory, f"{file_name.split('.')[0]}_result.jpg")
            mmcv.imwrite(img, save_image_path)

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
