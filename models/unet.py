import torch
import torch.nn as nn
import torch.nn.functional as F

# ConvBlock
class ConvBlock(nn.Module): 
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential( 
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    

# Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        return x, p
    
# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:])

        x = torch.cat([skip, x], dim=1)

        x = self.conv(x)

        return x
    

# U-Net
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=3):
        super().__init__()

        self.e1 = EncoderBlock(in_channels, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)

        self.b = ConvBlock(256, 512)

        self.d1 = DecoderBlock(512, 256)
        self.d2 = DecoderBlock(256, 128)
        self.d3 = DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b = self.b(p3)

        d1 = self.d1(b, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        return self.out(d3)