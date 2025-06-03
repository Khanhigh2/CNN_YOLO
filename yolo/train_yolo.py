from ultralytics import YOLO
import os

os.chdir("dataset/dataset_yolo")
model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=50, imgsz=640, batch=8)
