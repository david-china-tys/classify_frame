import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary
from thop import profile


class bottleneck(nn.Module):
    # t:输入通道的倍增系数
    def __init__(self, in_ch, out_ch, t, stride):
        super(bottleneck, self).__init__()

        expansion = in_ch * t

        self.shorcut = True if stride == 1 and in_ch == out_ch else False
        layer = []
        if t != 1:
            # pw
            layer.extend([
                nn.Conv2d(in_ch, expansion, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expansion),
                nn.ReLU6(inplace=True),
            ])
        # dw_linear
        layer.extend([
            nn.Conv2d(expansion, expansion, kernel_size=3, stride=stride, padding=1, groups=expansion, bias=False),
            nn.BatchNorm2d(expansion),
            nn.ReLU6(inplace=True),
            nn.Conv2d(expansion, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
        ])

        self.CONV = nn.Sequential(*layer)

    def forward(self, x):
        if self.shorcut:
            return x + self.CONV(x)
        else:
            return self.CONV(x)


# num:重复次数
class Block(nn.Module):
    def __init__(self, in_ch, out_ch, t, stride, num):
        super(Block, self).__init__()
        self.out_ch = out_ch
        layers = []
        for i in range(num):
            s = stride if i == 0 else 1
            layers.append(bottleneck(in_ch, out_ch, t, s))
            in_ch = out_ch
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Mobilenetv2(nn.Module):
    def __init__(self, num_classes=1000):
        super(Mobilenetv2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            Block(32, 16, 1, 1, 1),
            Block(16, 24, 6, 2, 2),
            Block(24, 32, 6, 2, 3),
            Block(32, 64, 6, 2, 4),
            Block(64, 96, 6, 1, 3),
            Block(96, 160, 6, 2, 3),
            Block(160, 320, 6, 1, 1),
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(1280, num_classes, kernel_size=1)

        )
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(self.net(x), 1)
        return self.fc(x)


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Mobilenetv2().to(device)
    #model = models.MobileNetV2().to(device)
    print(model)
    inputs = torch.ones([1, 3, 224, 224]).to(device)
    inputs_size = (3, 224, 224)
    summary(model, inputs_size)

    flops, params = profile(model=model, inputs=(inputs,))
    print('Model: {:.2f} GFLOPs and {:.2f}M parameters'.format(flops / 1e9, params / 1e6))
