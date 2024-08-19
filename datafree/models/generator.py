import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class Flatten(nn.Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return paddle.flatten(x, start_axis=1)

class Generator(nn.Layer):
    def __init__(self, nz=100, ngf=64, img_size=32, nc=3):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(nz, ngf * 2 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2D(ngf * 2),
            nn.Upsample(scale_factor=2),

            nn.Conv2D(ngf*2, ngf*2, 3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(ngf*2),
            nn.LeakyReLU(0.2),  # 移除了 inplace 参数
            nn.Upsample(scale_factor=2),

            nn.Conv2D(ngf*2, ngf, 3, stride=1, padding=1, bias_attr=False),
            nn.BatchNorm2D(ngf),
            nn.LeakyReLU(0.2),  # 移除了 inplace 参数
            nn.Conv2D(ngf, nc, 3, stride=1, padding=1),
            nn.Sigmoid(),  
        )

    def forward(self, z):
        out = self.l1(z)
        out = paddle.reshape(out, [out.shape[0], -1, self.init_size, self.init_size])
        img = self.conv_blocks(out)
        return img
