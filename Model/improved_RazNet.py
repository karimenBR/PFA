import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel = self.channel_attention(x)
        x = x * channel
        spatial = self.spatial_attention(x)
        return x * spatial

class ImprovedResNet(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ImprovedResNet, self).__init__()

        # Load pre-trained ResNet18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.resnet = models.resnet18(weights=weights)

        # Enhanced input processing for grayscale images
        self.input_processor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        # Initialize new conv1 weights properly
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            # Average pretrained weights across RGB channels for grayscale
            with torch.no_grad():
                self.resnet.conv1.weight[:,0] = original_conv1.weight.mean(dim=1)

        # Add attention modules after each residual block
        self.cbam1 = CBAM(64)
        self.cbam2 = CBAM(128)
        self.cbam3 = CBAM(256)
        self.cbam4 = CBAM(512)

        # Enhanced classifier head
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        # Initialize weights for new layers
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input processing
        x = self.input_processor(x)

        # Forward through ResNet with attention
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.cbam1(x)

        x = self.resnet.layer2(x)
        x = self.cbam2(x)

        x = self.resnet.layer3(x)
        x = self.cbam3(x)

        x = self.resnet.layer4(x)
        x = self.cbam4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

# Example usage
if __name__ == "__main__":
    model = ImprovedResNet(num_classes=14)
    print(model)

    # Test forward pass
    x = torch.randn(2, 1, 224, 224)  # Batch of 2 grayscale 224x224 images
    out = model(x)
    print(f"Output shape: {out.shape}")  # Should be [2, 10] for 10 classes