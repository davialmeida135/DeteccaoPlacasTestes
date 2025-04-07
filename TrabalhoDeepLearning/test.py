# test2.py

import cv2
from torch.utils.data import DataLoader
from dataloader import YOLODataset  # Ensure this links correctly
import torch
from model2 import ResNetYOLODetector  # Ensure this matches your model architecture
import os
import numpy as np
# At the top of your test.py file or in an appropriate section
from collections import namedtuple

Detection = namedtuple('Detection', ['box', 'score', 'class_idx'])

def load_model(model_path, device='cuda'):
    # Initialize the model architecture
    model = ResNetYOLODetector(
        num_classes=38,
        image_size=640,
        num_anchors=3
    )

    checkpoint = torch.load(model_path, map_location=device)

    # Load the saved weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model

# test2.py (continued)

def get_test_loader(img_dir, annot_dir, batch_size=1, num_workers=0, pin_memory=True):
    test_dataset = YOLODataset(
        img_dir=img_dir,
        annot_dir=annot_dir,
        transform=None  # Add any necessary transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: tuple(zip(*x))  # Use the appropriate collate function
    )
    
    return test_loader

def decode_bbox(bbox, grid_size, anchor_scales, image_size):
    """
    Decode bounding boxes from model predictions to absolute image coordinates.

    Args:
        bbox (np.array): Bounding box coordinates [batch_size, total_anchors, 4].
        grid_size (int): Size of the grid for the current scale.
        anchor_scales (list): List of anchor scales for the current scale.
        image_size (int): Size of the input image (assuming square images).

    Returns:
        np.array: Decoded bounding boxes [batch_size, total_anchors, 4] in [x_min, y_min, x_max, y_max].
    """
    batch_size, total_anchors, _ = bbox.shape
    decoded_boxes = np.zeros_like(bbox)

    stride = image_size / grid_size

    for i in range(batch_size):
        for j in range(total_anchors):
            cx, cy, w, h = bbox[i, j]

            # Calculate center coordinates
            cx_absolute = cx * stride
            cy_absolute = cy * stride

            # Calculate width and height
            anchor_w, anchor_h = anchor_scales[j % len(anchor_scales)]
            w_absolute = np.exp(w) * anchor_w
            h_absolute = np.exp(h) * anchor_h

            # Calculate corner coordinates
            x_min = cx_absolute - (w_absolute / 2)
            y_min = cy_absolute - (h_absolute / 2)
            x_max = cx_absolute + (w_absolute / 2)
            y_max = cy_absolute + (h_absolute / 2)

            # Clamp coordinates to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image_size, x_max)
            y_max = min(image_size, y_max)

            decoded_boxes[i, j] = [x_min, y_min, x_max, y_max]

    return decoded_boxes

def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) of two boxes.

    Args:
        box1 (list or np.array): [x_min, y_min, x_max, y_max].
        box2 (list or np.array): [x_min, y_min, x_max, y_max].

    Returns:
        float: IoU value.
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def nms(detections, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression to filter overlapping boxes.

    Args:
        detections (list): List of Detection namedtuples.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        list: Filtered detections after NMS.
    """
    if not detections:
        return []

    # Sort detections by score in descending order
    detections = sorted(detections, key=lambda x: x.score, reverse=True)
    keep = []

    while detections:
        current = detections.pop(0)
        keep.append(current)
        detections = [
            det for det in detections
            if compute_iou(current.box, det.box) < iou_threshold or det.class_idx != current.class_idx
        ]

    return keep

classes = []
for i in range(38):
    classes.append(f'class{i}')
def run_inference(model, image, device, image_size=640, class_names=classes):
    """
    Perform inference on a single OpenCV image, draw bounding boxes, and visualize the result.

    Args:
        model (torch.nn.Module): Trained object detection model.
        image (np.array): Input image in BGR format.
        device (torch.device): Device to perform inference on.
        image_size (int, optional): Size to which the image is resized. Defaults to 640.
        class_names (list, optional): List of class names. Defaults to ['class1', 'class2', 'class3'].
    """
    all_predictions = []
    all_targets = []
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image_resized = cv2.resize(image_rgb, (image_size, image_size))
    
    # Normalize image
    image_normalized = image_resized / 255.0  # Assuming model expects [0,1] range
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions
    batch_preds = []
    batch_scores = []
    batch_classes = []
    
    anchor_scales = [
        [(10, 13), (16, 30), (33, 23)],    # Scale 0
        [(30, 61), (62, 45), (59, 119)],   # Scale 1
        [(116, 90), (156, 198), (373, 326)] # Scale 2
    ]
    
            # Iterate over each scale's predictions
    for scale_idx, prediction in enumerate(predictions):
        # prediction shape: [batch_size, grid_h, grid_w, anchors, 43]
        batch_size, grid_h, grid_w, anchors, attrs = prediction.shape
        print(f"Scale {scale_idx}: Grid Size = {grid_h}x{grid_w}, Anchors = {anchors}, Attributes = {attrs}")
        
        # Split attributes
        bbox = prediction[..., :4]          # x, y, w, h
        objectness = prediction[..., 4]     # Objectness score
        class_scores = prediction[..., 5:]  # Class probabilities
        
        # Apply sigmoid to objectness and class scores
        objectness = torch.sigmoid(objectness)
        class_scores = torch.softmax(class_scores, dim=-1)
        
        # Reshape for easier processing
        bbox = bbox.contiguous().view(batch_size, -1, 4)          # [batch_size, grid_h * grid_w * anchors, 4]
        objectness = objectness.contiguous().view(batch_size, -1) # [batch_size, grid_h * grid_w * anchors]
        class_scores = class_scores.contiguous().view(batch_size, -1, class_scores.shape[-1]) # [batch_size, grid_h * grid_w * anchors, num_classes]
        
        batch_preds.append(bbox)
        batch_scores.append(class_scores)
    
    # Concatenate predictions from all scales
    bbox_coords = torch.cat(batch_preds, dim=1)    # [batch_size, total_anchors, 4]
    class_scores = torch.cat(batch_scores, dim=1)  # [batch_size, total_anchors, num_classes]
    # Convert predictions to desired format
    # Convert predictions to desired format
    for i in range(1):
        preds = bbox_coords[i].cpu().numpy()
        scores = class_scores[i].cpu().numpy()
        
        # Example threshold values
        score_threshold = 0.5
        nms_threshold = 0.5

        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []

        # Filter based on score thresholds
        for box, score in zip(preds, scores):
            class_idx = np.argmax(score)
            class_score = score[class_idx]
            if class_score >= score_threshold:
                filtered_boxes.append(box)
                filtered_scores.append(class_score)
                filtered_classes.append(class_idx)
        
        # Create Detection namedtuples
        detections = [
            Detection(box=box, score=score, class_idx=cls)
            for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes)
        ]
        
        # Apply Non-Maximum Suppression
        final_detections = nms(detections, iou_threshold=nms_threshold)
        
        # Extract final boxes, scores, and classes
        final_boxes = [det.box for det in final_detections]
        final_scores = [det.score for det in final_detections]
        final_classes = [det.class_idx for det in final_detections]
        print("Predicted Bounding boxes:", final_boxes)
        
        all_predictions.append((final_boxes, final_scores, final_classes))
        

        # print("Ground Truth Bounding boxes:", gt_boxes)

    return all_predictions
'''    # Process each image in the batch (only one image in this case)
    preds = bbox_coords[0]
    scores = class_scores[0]
    objs = objectness[0]
    
    filtered_boxes = []
    filtered_scores = []
    filtered_classes = []
    
    for j in range(len(objs)):
        if objs[j] < 0.5:
            continue
        class_idx = np.argmax(scores[j])
        class_score = scores[j][class_idx]
        if class_score < 0.5:
            continue
        filtered_boxes.append(preds[j])
        filtered_scores.append(objs[j] * class_score)
        filtered_classes.append(class_idx)
    
    detections = [
        Detection(box=box, score=score, class_idx=cls)
        for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes)
    ]
    
    final_detections = nms(detections, iou_threshold=0.5)
    
    final_boxes = [det.box for det in final_detections]
    final_scores = [det.score for det in final_detections]
    final_classes = [det.class_idx for det in final_detections]
    
    # Draw bounding boxes on the original image
    for box, score, cls in zip(final_boxes, final_scores, final_classes):
        x_min, y_min, x_max, y_max = box
        # Scale coordinates back to original image size
        scale_x = image.shape[1] / image_size
        scale_y = image.shape[0] / image_size
        x_min = int(x_min * scale_x)
        y_min = int(y_min * scale_y)
        x_max = int(x_max * scale_x)
        y_max = int(y_max * scale_y)
        
        # Draw rectangle
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Put label
        label = f"{class_names[cls]}: {score:.2f}"
        cv2.putText(image, label, (x_min, y_min - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    print(final_boxes)
    
    # Display the image
    cv2.imshow('Detections', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return final_boxes, final_scores, final_classes'''

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'best_model.pth'  # Adjust the path if necessary
    model = load_model(model_path, device)
    model.eval()
    print("Model loaded successfully.")

    # Read an image using OpenCV
    image_path = 'dataset/test/images/0e08e75d-6304-4c83-a7b2-4d97458838e2_jpg.rf.66d2bd1aed9f770d1b6cb0c28d565cc3.jpg'
    image = cv2.imread(image_path)

    # Define your class names

    # Perform inference and visualize
    final_boxes, final_scores, final_classes = run_inference(model, image, device, image_size=640)