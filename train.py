# Truong Dai hoc Cong nghe TP.HCM
#Tran Thi Yen Nhi - 2186300543 - 21DRTA1
#Training

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='dataset.yaml', epochs=50, imgsz=640)