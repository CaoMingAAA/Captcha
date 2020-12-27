import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, in_channel=3, out_channel=512, num_class=62, num_char=4):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=3, padding=(1, 1)),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(in_channels=16,out_channels= 64, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # batch*512*15*5
            nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=3, padding=(1, 1)),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        )
        # batch*512*7*2
        self.linear = nn.Linear(512 * 7 * 2, num_class * num_char)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 512 * 7 * 2)
        x = self.linear(x)
        return x
