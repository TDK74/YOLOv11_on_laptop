import cv2
from time import time
#import torch
from ultralytics import YOLO


# load YOLO11n
# model = YOLO('yolo11n.pt')
# load YOLO11s
# model = YOLO('yolo11s.pt')
# load YOLO11m
model = YOLO('yolo11m.pt')
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error opening camera.")
    exit()

width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps = 0

def fps_calculation(s_time, e_time):
    """
    Calculate frames per second.
    """
    loop_time = e_time - s_time
    fps_func = 1 / loop_time

    return fps_func


def objects_processing(framework):
    """
    Process detected objects and draw bounding boxes on the frame.
    """
    results = model(framework, device='cpu')

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box[ : 4])
            confidence = confidences[i]
            class_id = class_ids[i]
            label = result.names[class_id]

            cv2.rectangle(framework, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(framework, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    return framework


try:
    while True:
        start_time = time()
        ret, frame = cam.read()

        if not ret:
            print("Error reading frame.")
            break

        frame = objects_processing(frame)

        end_time = time()
        fps = fps_calculation(start_time, end_time)

        cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        #cv2.imshow('YOLO11n Real-time', frame)
        #cv2.imshow('YOLO11s Real-time', frame)
        cv2.imshow('YOLO11m Real-time', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("User's Ctrl+C detected.")

finally:
    cam.release()
    cv2.destroyAllWindows()
