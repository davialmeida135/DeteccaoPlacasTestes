import os

from ultralytics import YOLO
#LEVOU UMA HORA
def main():
    # Load a model
    model = YOLO("yolo11n.yaml")  # build a new model from scratch

    # Use the model
    results = model.train(data=os.path.join("D:\Github\DeteccaoPlacasTestes\Characters_South_America.v2i.yolov9\data.yaml"), epochs=50)  # train the model

if __name__ == "__main__":
    main()