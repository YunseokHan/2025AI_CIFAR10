import torch
from torchvision import datasets, transforms
import random

class CIFAR10Variant(torch.utils.data.Dataset):
    def __init__(self, root, train, transform, variant='baseline', noise_ratio=0.2):
        self.ds = datasets.CIFAR10(root, train=train, download=True, transform=transform)
        self.variant = variant
        self.noise_ratio = noise_ratio
        if train:
            self._apply_label_variant()
            if variant == 'input_perturb':
                self._wrap_input_transform()

    def _apply_label_variant(self):
        if self.variant == 'random_shuffle':
            random.shuffle(self.ds.targets)
        elif self.variant == 'label_noise':
            n = int(len(self.ds.targets) * self.noise_ratio)
            idx = random.sample(range(len(self.ds.targets)), n)
            for i in idx:
                self.ds.targets[i] = random.randrange(10)

    def _wrap_input_transform(self):
        extra = transforms.Compose([
            transforms.GaussianBlur(kernel_size=5),
            transforms.RandomAdjustSharpness(sharpness_factor=2)
        ])
        orig = self.ds.transform
        self.ds.transform = transforms.Compose([orig, extra])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds[idx]