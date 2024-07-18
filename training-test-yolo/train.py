import os

from ultralytics import YOLO
#LEVOU UMA HORA

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data=os.path.join("D:\Github\DeteccaoPlacas\Characters_South_America.v2i.yolov9\data.yaml"), epochs=1)  # train the model

