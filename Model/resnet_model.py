import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights  # Import weights enum


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet18 with updated weights parameter
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Modify the first convolutional layer to accept 1 channel (grayscale) instead of 3 (RGB)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final fully connected layer to match the number of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)