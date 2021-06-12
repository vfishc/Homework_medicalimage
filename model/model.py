import torch
import torch.nn as nn

class fracnet(nn.Module):
    def __init__(self,start_channels,end_channels,mid_channels = 16):
        super().__init__()
        self.pre = new_convolution(start_channels,mid_channels)
        start_channels = mid_channels
        self.encode1 = encoder(start_channels,2*start_channels)
        self.encode2 = encoder(2*start_channels,4*start_channels)
        self.encode3 = encoder(4*start_channels,8*start_channels)
        self.decode3 = decoder(8*start_channels,4*start_channels)
        self.decode2 = decoder(4*start_channels,2*start_channels)
        self.decode1 = decoder(2*start_channels,start_channels)
        self.output = nn.Conv3d(start_channels,end_channels,1)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


    def forward(self,x):
        pre_encode = self.pre(x)
        encoding1 = self.encode1(pre_encode)
        encoding2 = self.encode2(encoding1)
        encoding3 = self.encode3(encoding2)
        x = self.decode3(encoding3,encoding2)
        x = self.decode2(x,encoding1)
        x = self.decode1(x,pre_encode)
        x = self.output(x)
        return x




class new_convolution(nn.Sequential):
    def __init__(self, start_channels, end_channels):
        super().__init__(
            nn.Conv3d(start_channels, end_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(end_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(end_channels, end_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(end_channels),
            nn.LeakyReLU(inplace=True))

class encoder(nn.Sequential):
    def __init__(self, start_channels, end_channels):
        super().__init__(nn.MaxPool3d(2),new_convolution(start_channels, end_channels))

class decoder(nn.Module):
    def __init__(self,start_channels,end_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(start_channels, end_channels, 2, stride=2, bias=False),
            nn.BatchNorm3d(end_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = new_convolution(start_channels,end_channels)

    def forward(self,x,y):
        x = self.conv1(x)
        x = self.conv2(torch.cat([y, x], dim=1))
        return x


        