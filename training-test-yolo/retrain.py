import os
import pathlib
from ultralytics import YOLO
def main():
    self_path =pathlib.Path(__file__).parent.resolve()
    model_path = os.path.join(self_path, '..',"runs/detect/train8/weights/best.pt")

    # Load a model
    model = YOLO(model_path)  # build a new model from scratch

    # Use the model
    model.train(data=os.path.join(self_path,"..","Characters_South_America.v2i.yolov9/data.yaml"), epochs=100)  # train the model

if __name__ == "__main__":
    main()
