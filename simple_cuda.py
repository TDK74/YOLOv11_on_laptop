import ultralytics
from ultralytics import YOLO


model = YOLO('yolov8n.pt')  # load an official PyTorch model
# Enter your path to the file in the example path below
results = model(source='DiskName (C, D, E...):\path-to-folder\file_name.jpg', device='cuda')  # use of GPU for processing
