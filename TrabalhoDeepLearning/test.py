# test2.py


from torch.utils.data import DataLoader
from dataloader import YOLODataset  # Ensure this links correctly
import torch
from model2 import ResNetYOLODetector  # Ensure this matches your model architecture
import os

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

def run_inference(model, dataloader, device):
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            print(type(images))
            print(len(images))
            print(images[0].shape)
            images = torch.stack([image.to(device, dtype=torch.float32) for image in images]).to(device)
            
            # Forward pass
            predictions = model(images)
            print(type(predictions))
            print(len(predictions))
            print(predictions[0].shape)
            print(predictions[1].shape)
            print(predictions[2].shape)
            print(type(predictions[2]))
            # Process predictions
            # Initialize lists to store predictions for this batch
            batch_preds = []
            batch_scores = []
            
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
            for i in range(images.size(0)):
                preds = bbox_coords[i].cpu().numpy()
                scores = class_scores[i].cpu().numpy()
                all_predictions.append((preds, scores))
                
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                all_targets.append((gt_boxes, gt_labels))
                print("Bounding boxes:", gt_boxes)
                
    return all_predictions, all_targets

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = 'best_model.pth'  # Adjust the path if necessary
    model = load_model(model_path, device)
    print("Model loaded successfully.")
    img_dir = 'dataset/test/images'
    annot_dir = 'dataset/test/labels'
    
    test_loader = get_test_loader(img_dir, annot_dir)
    print("Test DataLoader initialized.")

    all_preds, all_tgts = run_inference(model, test_loader, device)
    print("Inference completed.")