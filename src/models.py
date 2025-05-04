import torchvision.models as models
import torch.nn as nn

def get_model(name='resnet18'):
    if name == 'resnet18':
        model = models.resnet18(weights='DEFAULT', num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    if name == 'resnet50':
        model = models.resnet50(weights='DEFAULT', num_classes=10)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    
    return model