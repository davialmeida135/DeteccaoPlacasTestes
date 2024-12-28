import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
from dataloader import YOLODataset
from model2 import ResNetYOLODetector
import os

def convert_targets_to_grid(batch_targets, grid_sizes, num_anchors, num_classes, image_size=640):
    """Convert raw targets to grid-based format"""
    batch_size = len(batch_targets)
    device = 'cuda'
    grid_targets = []
    
    for grid_size in grid_sizes:
        # Initialize targets for this scale
        scale_targets = torch.zeros(
            (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes),
            device=device
        )
        
        for b in range(batch_size):
            boxes = batch_targets[b]['boxes']
            labels = batch_targets[b]['labels']
            
            if len(boxes) == 0:
                continue
                
            # Convert boxes to grid coordinates
            grid_stride = image_size / grid_size
            x_centers = (boxes[:, 0] + boxes[:, 2]) / 2 / grid_stride
            y_centers = (boxes[:, 1] + boxes[:, 3]) / 2 / grid_stride
            widths = (boxes[:, 2] - boxes[:, 0]) / grid_stride
            heights = (boxes[:, 3] - boxes[:, 1]) / grid_stride
            
            # Get grid cell indices
            grid_x = x_centers.long().clamp(0, grid_size-1)
            grid_y = y_centers.long().clamp(0, grid_size-1)
            
            # Assign targets to grid cells
            for idx in range(len(boxes)):
                # Verify that label is within valid range
                if labels[idx] >= num_classes:
                    print(f"Warning: Found label {labels[idx]} which is >= num_classes ({num_classes})")
                    continue

                # Assign to best anchor (simplified)
                anchor_idx = idx % num_anchors
                
                # Set target values
                scale_targets[b, grid_y[idx], grid_x[idx], anchor_idx, 0:4] = torch.tensor(
                    [x_centers[idx] - grid_x[idx], y_centers[idx] - grid_y[idx], 
                     widths[idx], heights[idx]]
                )
                scale_targets[b, grid_y[idx], grid_x[idx], anchor_idx, 4] = 1.0  # objectness
                
                # Ensure class label is within bounds
                class_idx = labels[idx].item()
                if 0 <= class_idx < num_classes:
                    scale_targets[b, grid_y[idx], grid_x[idx], anchor_idx, 5 + class_idx] = 1.0
                
        grid_targets.append(scale_targets)
    
    return grid_targets

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)
    
    # Loss functions
    bbox_loss_fn = nn.MSELoss(reduction='mean')
    obj_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    cls_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        
        # Convert targets to grid format
        grid_targets = convert_targets_to_grid(
            targets, 
            grid_sizes=[80, 40, 20],
            num_anchors=3,
            num_classes=38,
            image_size=640
        )
        
        # Forward pass
        predictions = model(images)
        
        # Calculate loss for each scale
        loss = 0
        for pred, target in zip(predictions, grid_targets):
            # Bounding box loss
            bbox_loss = bbox_loss_fn(
                pred[..., :4][target[..., 4] == 1],
                target[..., :4][target[..., 4] == 1]
            )
            
            # Objectness loss
            obj_loss = obj_loss_fn(pred[..., 4], target[..., 4])
            
            # Classification loss
            cls_loss = cls_loss_fn(
                pred[..., 5:][target[..., 4] == 1],
                target[..., 5:][target[..., 4] == 1]
            )
            
            # Combine losses
            scale_loss = bbox_loss + obj_loss + cls_loss
            loss += scale_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    return total_loss / num_batches

def collate_fn(batch):
    """
    Custom collate function to handle variable number of boxes
    """
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
        
    images = torch.stack(images, 0)
    
    return images, targets

def main():
    # Dataset setup
    train_dataset = YOLODataset(
        img_dir='dataset/train/images',
        annot_dir='dataset/train/labels',
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Model setup
    device = 'cuda'
    model = ResNetYOLODetector(
        num_classes=38,
        image_size=640,
        num_anchors=3
    ).to(device)
    
    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # Training loop
    num_epochs = 10
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        
        # Update learning rate
        scheduler.step(train_loss)
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, 'best_model.pth')
            
        print(f'Epoch {epoch+1} - Loss: {train_loss:.4f}')
    
    save_dir = 'saved_models'
    # Save the final model after all epochs
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss,
    }, os.path.join(save_dir, 'final_model.pth'))
    print('\nTraining completed. Final model saved as final_model.pth')

if __name__ == '__main__':
    main()