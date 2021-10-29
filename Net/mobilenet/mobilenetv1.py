import torch
import torch.nn as nn


# 标准卷积-BN-relu
import torchvision.models


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super(Conv, self).__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=s, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv2d(x)


class DWconv(nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super(DWconv, self).__init__()
        self.conv_dw = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3, stride=s, padding=1, groups=in_ch,
                      bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_dw(x)


# 输入的尺寸大小224*224
class MobilenetV1(nn.Module):
    def __init__(self):
        super(MobilenetV1, self).__init__()
        self.feature = nn.Sequential(
            Conv(3, 32, 2),
            DWconv(32, 64, 1),
            DWconv(64, 128, 2),
            DWconv(128, 128, 1),
            DWconv(128, 256, 2),
            DWconv(256, 256, 1),
            DWconv(256, 512, 2),
            DWconv(512, 512, 1),
            DWconv(512, 512, 1),
            DWconv(512, 512, 1),
            DWconv(512, 512, 1),
            DWconv(512, 512, 1),
            DWconv(512, 1024, 2),
            DWconv(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.feature(x)
        #print(out.shape)

        out = torch.flatten(out, 1)
        #print(out.shape)

        out = self.fc(out)
        return out


if __name__ == '__main__':
    a = torch.randn(1, 3, 224, 224)
    net = MobilenetV1()
    res = net(a)
    from thop import profile

    print(res.shape)
    flops,params=profile(model=net,inputs=(a,))
    print('Model:{:.2f} GfLops and {:.2f}M parameters'.format(flops/1e9,params/1e6))
    #print(res)

    # b=torch.randn(4, 3)
    # print(b)
    # print(torch.softmax(b,1))