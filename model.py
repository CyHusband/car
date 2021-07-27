import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class Car(nn.Module):
    def __init__(self):
        super(Car, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        for p in self.parameters():
            p.requires_grad = False
        self.resnet.fc = nn.Linear(2048, 5, bias=True)

    def forward(self, x):
        out = self.resnet(x)
        return out
