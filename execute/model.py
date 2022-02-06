import execute.resnet as models
import torch.nn as nn
import torch
from efficientnet.model import EfficientNet

class model(nn.Module):

    def __init__(self, num_slices=10, embedding_size=64, num_classes=2, parallel=False):
        super().__init__()
        self.num_slices = num_slices
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.subnet = EfficientNet.from_name('efficientnet-b4', 1, num_classes=embedding_size)
        if parallel:
            self.subnet = nn.DataParallel(self.subnet)
        self.fc = nn.Linear(embedding_size * num_slices, num_classes)

    def _forward_impl(self, x):
        x = self.subnet(x)
        x = x.view(-1, self.num_slices * self.embedding_size)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)