import torchvision.models as models
import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

def get_model(name='resnet18'):
    if name == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        embed_dim = model.fc.weight.shape[1]
        model.fc = LinearClassifier(embed_dim, num_labels=10)
    
    if name == 'resnet50':
        model = models.resnet50(weights='DEFAULT')
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        embed_dim = model.fc.weight.shape[1]
        model.fc = LinearClassifier(embed_dim, num_labels=10)
    
    return model