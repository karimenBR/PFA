import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet34_Weights

class ResNet(nn.Module):
    def __init__(self, num_classes=14):  # ChestMNIST has 14 labels
        super(ResNet, self).__init__()
        self.resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)