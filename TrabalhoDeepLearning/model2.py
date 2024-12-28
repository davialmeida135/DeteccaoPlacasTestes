import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F  # This imports the functional module

class ResNetYOLODetector(nn.Module):
    def __init__(self, num_classes=27, image_size=640, num_anchors=3):
        super(ResNetYOLODetector, self).__init__()
        self.num_classes = num_classes
        self.image_size = image_size
        self.num_anchors = num_anchors
        
        # Calculate grid sizes for different scales
        self.grid_sizes = [image_size // 8, image_size // 16, image_size // 32]  # [80, 40, 20]
        
        # Base ResNet feature extractor
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Remove avg pool and fc
        
        # Feature Pyramid Network (FPN)
        self.fpn_channels = 256
        self.lateral_conv1 = nn.Conv2d(512, self.fpn_channels, 1)  # ResNet layer4
        self.lateral_conv2 = nn.Conv2d(256, self.fpn_channels, 1)  # ResNet layer3
        self.lateral_conv3 = nn.Conv2d(128, self.fpn_channels, 1)  # ResNet layer2
        
        # Detection heads for each scale
        self.detection_heads = nn.ModuleList([
            self._make_detection_head() for _ in range(3)
        ])
        
    def _make_detection_head(self):
        """Creates a detection head for a single scale"""
        return nn.Sequential(
            nn.Conv2d(self.fpn_channels, self.fpn_channels, 3, padding=1),
            nn.BatchNorm2d(self.fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fpn_channels, self.num_anchors * (5 + self.num_classes), 1)
        )
    
    def _upsample_add(self, x, y):
        """Upsample x and add it to y"""
        return F.interpolate(x, size=y.shape[2:], mode='nearest') + y

    def forward(self, x):
        # Extract features at different scales
        features = []
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [5, 6, 7]:  # Store intermediate features for FPN
                features.append(x)
        
        # FPN forward pass
        c3, c4, c5 = features
        p5 = self.lateral_conv1(c5)
        p4 = self._upsample_add(p5, self.lateral_conv2(c4))
        p3 = self._upsample_add(p4, self.lateral_conv3(c3))
        
        pyramid_features = [p3, p4, p5]
        
        # Detection heads
        outputs = []
        for feature, head, grid_size in zip(pyramid_features, self.detection_heads, self.grid_sizes):
            output = head(feature)
            
            # Reshape output: [batch, anchors * (5 + num_classes), grid, grid] ->
            #                [batch, grid, grid, anchors, (5 + num_classes)]
            batch_size = output.shape[0]
            output = output.view(batch_size, self.num_anchors, 5 + self.num_classes, 
                               grid_size, grid_size)
            output = output.permute(0, 3, 4, 1, 2)
            
            # Apply activation functions
            # First 4 values (bbox) -> sigmoid
            # 5th value (objectness) -> sigmoid
            # Remaining values (class scores) -> left as logits for CrossEntropyLoss
            output[..., :4] = torch.sigmoid(output[..., :4])
            output[..., 4] = torch.sigmoid(output[..., 4])
            
            outputs.append(output)
            
        return outputs
    
    def compute_grid_offsets(self, grid_size, device):
        """Compute grid cell offsets for decoding predictions"""
        grid_y, grid_x = torch.meshgrid(torch.arange(grid_size), torch.arange(grid_size))
        grid_xy = torch.stack((grid_x, grid_y), dim=-1).to(device)
        return grid_xy.view(1, grid_size, grid_size, 1, 2).float()
    
if __name__ == "__main__":
    # Initialize the model
    model = ResNetYOLODetector(
        num_classes=27,
        image_size=640,
        num_anchors=3
    )

    # Example forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 640, 640)
    outputs = model(input_tensor)

    # outputs is a list of 3 tensors, one for each scale
    for i, output in enumerate(outputs):
        print(f"Scale {i} output shape:", output.shape)