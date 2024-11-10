from torch import nn
from torchvision import models


class FaceLandmark(nn.Module):
    def __init__(self):
        super(FaceLandmark, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 98 * 2)

    def forward(self, x):
        return self.backbone(x)
