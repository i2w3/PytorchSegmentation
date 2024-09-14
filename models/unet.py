import torch
import torch.nn as nn

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=2)
    def forward(self, x, is_pool=True):
        if is_pool:
            x = self.pool(x)
        x = self.conv_relu(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv_relu = nn.Sequential(
                            nn.Conv2d(2*channels, channels, kernel_size=3, padding=1),
                            nn.BatchNorm2d(channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(channels, channels,  kernel_size=3, padding=1),
                            nn.BatchNorm2d(channels),
                            nn.ReLU(inplace=True)
            )
        self.upconv_relu = nn.Sequential(nn.ConvTranspose2d(channels, 
                                                            channels//2, 
                                                            kernel_size=3,
                                                            stride=2,padding=1,
                                                            output_padding=1),
                                         nn.ReLU(inplace=True)
            )
        
    def forward(self, x):
        x = self.conv_relu(x)
        x = self.upconv_relu(x)
        return x
    
class UNet(nn.Module):
    def __init__(self, num_classes:int = 2):
        super(UNet, self).__init__()
        self.down1 = Downsample(3, 64)
        self.down2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.down4 = Downsample(256, 512)
        self.down5 = Downsample(512, 1024)
        
        self.up = nn.Sequential(nn.ConvTranspose2d(1024, 512, kernel_size=3,stride=2,padding=1,output_padding=1),
                                nn.ReLU(inplace=True)
                                )
        
        self.up1 = Upsample(512)
        self.up2 = Upsample(256)
        self.up3 = Upsample(128)
        
        self.conv_2 = Downsample(128, 64)
        self.last = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x, is_pool=False)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        
        x5 = self.up(x5)
        
        x5 = torch.cat([x4, x5], dim=1)           # 32*32*1024
        x5 = self.up1(x5)                         # 64*64*256
        x5 = torch.cat([x3, x5], dim=1)           # 64*64*512  
        x5 = self.up2(x5)                         # 128*128*128
        x5 = torch.cat([x2, x5], dim=1)           # 128*128*256
        x5 = self.up3(x5)                         # 256*256*64
        x5 = torch.cat([x1, x5], dim=1)           # 256*256*128
        
        x5 = self.conv_2(x5, is_pool=False)       # 256*256*64
        
        x5 = self.last(x5)                        # 256*256*3
        return x5
    
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    return m



if __name__ == "__main__":
    from torchinfo import summary

    model = UNet()
    batch_size = 16
    summary(model, input_size=(batch_size, 3, 256, 256))
'''
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
UNet                                     [16, 2, 256, 256]         --
├─Downsample: 1-1                        [16, 64, 256, 256]        --
│    └─Sequential: 2-1                   [16, 64, 256, 256]        --
│    │    └─Conv2d: 3-1                  [16, 64, 256, 256]        1,792
│    │    └─ReLU: 3-2                    [16, 64, 256, 256]        --
│    │    └─Conv2d: 3-3                  [16, 64, 256, 256]        36,928
│    │    └─ReLU: 3-4                    [16, 64, 256, 256]        --
├─Downsample: 1-2                        [16, 128, 128, 128]       --
│    └─MaxPool2d: 2-2                    [16, 64, 128, 128]        --
│    └─Sequential: 2-3                   [16, 128, 128, 128]       --
│    │    └─Conv2d: 3-5                  [16, 128, 128, 128]       73,856
│    │    └─ReLU: 3-6                    [16, 128, 128, 128]       --
│    │    └─Conv2d: 3-7                  [16, 128, 128, 128]       147,584
│    │    └─ReLU: 3-8                    [16, 128, 128, 128]       --
├─Downsample: 1-3                        [16, 256, 64, 64]         --
│    └─MaxPool2d: 2-4                    [16, 128, 64, 64]         --
│    └─Sequential: 2-5                   [16, 256, 64, 64]         --
│    │    └─Conv2d: 3-9                  [16, 256, 64, 64]         295,168
│    │    └─ReLU: 3-10                   [16, 256, 64, 64]         --
│    │    └─Conv2d: 3-11                 [16, 256, 64, 64]         590,080
│    │    └─ReLU: 3-12                   [16, 256, 64, 64]         --
├─Downsample: 1-4                        [16, 512, 32, 32]         --
│    └─MaxPool2d: 2-6                    [16, 256, 32, 32]         --
│    └─Sequential: 2-7                   [16, 512, 32, 32]         --
│    │    └─Conv2d: 3-13                 [16, 512, 32, 32]         1,180,160
│    │    └─ReLU: 3-14                   [16, 512, 32, 32]         --
│    │    └─Conv2d: 3-15                 [16, 512, 32, 32]         2,359,808
│    │    └─ReLU: 3-16                   [16, 512, 32, 32]         --
├─Downsample: 1-5                        [16, 1024, 16, 16]        --
│    └─MaxPool2d: 2-8                    [16, 512, 16, 16]         --
│    └─Sequential: 2-9                   [16, 1024, 16, 16]        --
│    │    └─Conv2d: 3-17                 [16, 1024, 16, 16]        4,719,616
│    │    └─ReLU: 3-18                   [16, 1024, 16, 16]        --
│    │    └─Conv2d: 3-19                 [16, 1024, 16, 16]        9,438,208
│    │    └─ReLU: 3-20                   [16, 1024, 16, 16]        --
├─Sequential: 1-6                        [16, 512, 32, 32]         --
│    └─ConvTranspose2d: 2-10             [16, 512, 32, 32]         4,719,104
│    └─ReLU: 2-11                        [16, 512, 32, 32]         --
├─Upsample: 1-7                          [16, 256, 64, 64]         --
│    └─Sequential: 2-12                  [16, 512, 32, 32]         --
│    │    └─Conv2d: 3-21                 [16, 512, 32, 32]         4,719,104
│    │    └─ReLU: 3-22                   [16, 512, 32, 32]         --
│    │    └─Conv2d: 3-23                 [16, 512, 32, 32]         2,359,808
│    │    └─ReLU: 3-24                   [16, 512, 32, 32]         --
│    └─Sequential: 2-13                  [16, 256, 64, 64]         --
│    │    └─ConvTranspose2d: 3-25        [16, 256, 64, 64]         1,179,904
│    │    └─ReLU: 3-26                   [16, 256, 64, 64]         --
├─Upsample: 1-8                          [16, 128, 128, 128]       --
│    └─Sequential: 2-14                  [16, 256, 64, 64]         --
│    │    └─Conv2d: 3-27                 [16, 256, 64, 64]         1,179,904
│    │    └─ReLU: 3-28                   [16, 256, 64, 64]         --
│    │    └─Conv2d: 3-29                 [16, 256, 64, 64]         590,080
│    │    └─ReLU: 3-30                   [16, 256, 64, 64]         --
│    └─Sequential: 2-15                  [16, 128, 128, 128]       --
│    │    └─ConvTranspose2d: 3-31        [16, 128, 128, 128]       295,040
│    │    └─ReLU: 3-32                   [16, 128, 128, 128]       --
├─Upsample: 1-9                          [16, 64, 256, 256]        --
│    └─Sequential: 2-16                  [16, 128, 128, 128]       --
│    │    └─Conv2d: 3-33                 [16, 128, 128, 128]       295,040
│    │    └─ReLU: 3-34                   [16, 128, 128, 128]       --
│    │    └─Conv2d: 3-35                 [16, 128, 128, 128]       147,584
│    │    └─ReLU: 3-36                   [16, 128, 128, 128]       --
│    └─Sequential: 2-17                  [16, 64, 256, 256]        --
│    │    └─ConvTranspose2d: 3-37        [16, 64, 256, 256]        73,792
│    │    └─ReLU: 3-38                   [16, 64, 256, 256]        --
├─Downsample: 1-10                       [16, 64, 256, 256]        --
│    └─Sequential: 2-18                  [16, 64, 256, 256]        --
│    │    └─Conv2d: 3-39                 [16, 64, 256, 256]        73,792
│    │    └─ReLU: 3-40                   [16, 64, 256, 256]        --
│    │    └─Conv2d: 3-41                 [16, 64, 256, 256]        36,928
│    │    └─ReLU: 3-42                   [16, 64, 256, 256]        --
├─Conv2d: 1-11                           [16, 2, 256, 256]         130
==========================================================================================
Total params: 34,513,410
Trainable params: 34,513,410
Non-trainable params: 0
Total mult-adds (Units.TERABYTES): 1.05
==========================================================================================
Input size (MB): 12.58
Forward/backward pass size (MB): 5117.05
Params size (MB): 138.05
Estimated Total Size (MB): 5267.69
==========================================================================================
'''