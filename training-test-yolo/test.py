import os
from ultralytics import YOLO
import cv2

# Load the saved model
script_dir = os.path.dirname(__file__)
model_path = os.path.join(script_dir, "../runs/detect/train3/weights/best.pt")
model = YOLO(model_path)

#test_images_dir = os.path.join(script_dir, "../Characters_South_America.v2i.yolov9/test/images")
test_images_dir = os.path.join(script_dir, "../testimages")
output_dir = os.path.join(script_dir, "../Characters_South_America.v2i.yolov9/test/output")
os.makedirs(output_dir, exist_ok=True)

# Get a list of test images
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

# Run predictions and save results
for img_path in test_images:
    # Run prediction
    results = model.predict(img_path,conf=0.15)
    
    # Debugging: print the results
    print(f"Predictions for {img_path}:")
    #for result in results:
    #    print(result)

    # Load the image
    img = cv2.imread(img_path)

    # Iterate over each result
    for result in results:
        if hasattr(result, 'boxes'):
            boxes = result.boxes  # Get the bounding boxes
            sorted_boxes = sorted(boxes, key=lambda box: box.xyxy[0][0])
            for box in sorted_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to integer coordinates
                confidence = box.conf[0]
                class_id = box.cls[0]
                label = f'{model.names[int(class_id)]} {confidence:.2f}'
                print("Label:",label, "x1:",x1,)

                # Draw the bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw the label
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            print(f"No bounding boxes found in the result for {img_path}")

    # Save the image
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, img)
