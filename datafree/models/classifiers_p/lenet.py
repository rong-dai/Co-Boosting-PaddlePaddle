import paddle
import x2paddle
import paddle.nn as nn

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class LeNet5(nn.Layer):

    def __init__(self, nc=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2D(1, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2, stride=2),
            nn.Conv2D(16, 120, kernel_size=5),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes)
        )

    def forward(self, img, return_features=False):
        features = self.features(img).reshape([img.shape[0], -1])
        output = self.fc(features)
        if return_features:
            return output, features
        return output
