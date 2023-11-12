from mmdet.apis import inference_detector
from utils import get_model
import os
import mmcv
import cv2
import time

def process_video(input_path='input/video_1.mp4', weights='yolov3_mobilenetv2_320_300e_coco', threshold=0.5, config_file='your_default_config_file_path.py'):
    # Build the model.
    model = get_model(config_file, weights)
    cap = mmcv.VideoReader(input_path)
    save_name = f"{input_path.split('/')[-1].split('.')[0]}_{weights}"
    
    output_dir='/data/operati123/yolo/outputs'
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        f"{output_dir}/flying.mp4", fourcc, cap.fps,
        (cap.width, cap.height)
    )
    
    frame_count = 0  # To count total frames.
    total_fps = 0  # To get the final frames per second.

    for frame in mmcv.track_iter_progress(cap):
        # Increment frame count.
        frame_count += 1
        start_time = time.time()  # Forward pass start time.
        result = inference_detector(model, frame)
        end_time = time.time()  # Forward pass end time.
        # Get the fps.
        fps = 1 / (end_time - start_time)
        # Add fps to total fps.
        total_fps += fps
        show_result = model.show_result(frame, result, score_thr=threshold)
        # Write the FPS on the current frame.
        cv2.putText(
            show_result, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 0, 255), 2, cv2.LINE_AA
        )
        out.write(show_result)

    # Release VideoWriter()
    out.release()
    
    # Calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


weights_path = '/data/operati123/yolo/model/epoch_15.pth'
config_file_path='./yolo_config_test.py'

# 사용 예시
process_video(input_path='/local_datasets/flying/flying.mp4',weights='/data/operati123/yolo/model/epoch_15.pth', threshold=0.5,config_file=config_file_path)

