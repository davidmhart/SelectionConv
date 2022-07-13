import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
from torch import nn

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.MaxPool2d(kernel_size=2, stride=2),
            _ResBlock(in_channels,in_channels,out_channels)
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = _ResBlock(2*in_channels, out_channels, out_channels)

    def forward(self, x, skip_connection):
        x = F.interpolate(x, scale_factor=2, mode='nearest') # Will be replaced with unpoolCluster
        x = torch.cat((x,skip_connection),dim=1)
        return self.decode(x)
    
class _ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3,padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.net(x) + self.res(x))

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels=3, feat=32):
        super(UNet, self).__init__()
        self.start = nn.Conv2d(in_channels, feat, kernel_size=3, padding=1)
        self.enc1 = _EncoderBlock(feat, feat*2)
        self.enc2 = _EncoderBlock(feat*2, feat*4)
        self.enc3 = _EncoderBlock(feat*4, feat*8)
        self.enc4 = _EncoderBlock(feat*8, feat*16)
        self.enc5 = _EncoderBlock(feat*16, feat*16)
        self.dec5 = _DecoderBlock(feat*16, feat*8)
        self.dec4 = _DecoderBlock(feat*8, feat*4)
        self.dec3 = _DecoderBlock(feat*4, feat*2)
        self.dec2 = _DecoderBlock(feat*2, feat)
        self.dec1 = _DecoderBlock(feat,feat)
        self.final = nn.Conv2d(feat, num_classes, kernel_size=3, padding=1)
        initialize_weights(self)

    def forward(self, x):
        enc1 = self.start(x)
        enc2 = self.enc1(enc1)
        enc3 = self.enc2(enc2)
        enc4 = self.enc3(enc3)
        enc5 = self.enc4(enc4)
        center = self.enc5(enc5)
        dec5 = self.dec5(center,enc5)
        dec4 = self.dec4(dec5,enc4)
        dec3 = self.dec3(dec4,enc3)
        dec2 = self.dec2(dec3,enc2)
        dec1 = self.dec1(dec2,enc1)
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='bilinear')#, [enc1,enc2,enc3,enc4,enc5,center,dec5,dec4,dec3,dec2,dec1] 
        # Interpolate only needed in 2D version to match odd pixel sizes, graphs will not need this interpolation