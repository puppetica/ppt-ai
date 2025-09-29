from torch import nn


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1),
            nn.BatchNorm2d(base_ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, 1, 4, 1, 1),  # patch output
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
