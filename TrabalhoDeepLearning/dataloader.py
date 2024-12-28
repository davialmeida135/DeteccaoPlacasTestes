import os
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import torchvision.transforms as transforms

class YOLODataset(Dataset):
    def __init__(self, img_dir, annot_dir, transform=None):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        # Define transforms directly in the class
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        annot_path = os.path.join(self.annot_dir, self.img_files[idx].replace('.jpg', '.txt'))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0, 1]
        height, width, _ = img.shape

        # Load annotations
        boxes = []
        labels = []
        with open(annot_path, 'r') as f:
            for line in f.readlines():
                cls, x_center, y_center, box_width, box_height = map(float, line.strip().split())
                x_min = (x_center - box_width / 2) * width
                y_min = (y_center - box_height / 2) * height
                x_max = (x_center + box_width / 2) * width
                y_max = (y_center + box_height / 2) * height
                boxes.append([x_min, y_min, x_max, y_max])
                labels.append(int(cls))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        # Apply transformations (if any)
        if self.transform:
            augmented = self.transform(image=img, bboxes=boxes, class_labels=labels)
            img = augmented['image']
            boxes = torch.tensor(augmented['bboxes'], dtype=torch.float32)
            labels = torch.tensor(augmented['class_labels'], dtype=torch.int64)

        # Return the image and the target dictionary
        return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), {'boxes': boxes, 'labels': labels}
