import cv2

def save_frames(video_path, output_path="./test/"):
    # 영상 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 저장될 이미지의 파일명 형식 지정
    filename_format = "{}{:06d}.jpg"

    # 프레임 단위로 이미지 저장
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 이미지 저장
        filename = filename_format.format(output_path, frame_count + 1)
        cv2.imwrite(filename, frame)

        frame_count += 1

    # 영상 파일 닫기
    cap.release()

if __name__ == "__main__":
    # 입력 영상 파일 경로 지정
    video_path = "./bird2.mp4"

    # 함수 호출
    save_frames(video_path)
