import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with skip connections
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.double_conv(in_channels, 64)
        self.enc2 = self.double_conv(64, 128)
        self.enc3 = self.double_conv(128, 256)
        self.enc4 = self.double_conv(256, 512)
        
        # Attention Gates
        self.att3 = AttentionGate(512, 256)
        self.att2 = AttentionGate(256, 128)
        self.att1 = AttentionGate(128, 64)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder with attention
        dec3 = self.up3(enc4)
        att3 = self.att3(dec3, enc3)
        dec3 = torch.cat([dec3, att3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        att2 = self.att2(dec2, enc2)
        dec2 = torch.cat([dec2, att2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        att1 = self.att1(dec1, enc1)
        dec1 = torch.cat([dec1, att1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_l)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_l, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResUNet, self).__init__()
        
        # Encoder
        self.enc1 = ResBlock(in_channels, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        self.enc4 = ResBlock(256, 512)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResBlock(128, 64)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class DenseUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, growth_rate=32):
        super(DenseUNet, self).__init__()
        
        # Encoder
        self.enc1 = DenseBlock(in_channels, 64, growth_rate)
        self.enc2 = DenseBlock(64, 128, growth_rate)
        self.enc3 = DenseBlock(128, 256, growth_rate)
        self.enc4 = DenseBlock(256, 512, growth_rate)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DenseBlock(512, 256, growth_rate)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DenseBlock(256, 128, growth_rate)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DenseBlock(128, 64, growth_rate)
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.conv3 = nn.Conv2d(growth_rate * 2, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out = torch.cat([out1, out2], dim=1)
        out = F.relu(self.bn3(self.conv3(out)))
        return out

class SEUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(SEUNet, self).__init__()
        
        # Encoder
        self.enc1 = SEBlock(self.double_conv(in_channels, 64))
        self.enc2 = SEBlock(self.double_conv(64, 128))
        self.enc3 = SEBlock(self.double_conv(128, 256))
        self.enc4 = SEBlock(self.double_conv(256, 512))
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = SEBlock(self.double_conv(512, 256))
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = SEBlock(self.double_conv(256, 128))
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = SEBlock(self.double_conv(128, 64))
        
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(2)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Decoder
        dec3 = self.up3(enc4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

class SEBlock(nn.Module):
    def __init__(self, module):
        super(SEBlock, self).__init__()
        self.module = module
        channels = module[-2].num_features
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.module(x)
        se_weight = self.se(x)
        return x * se_weight 

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super().__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Backbone (ResNet-like)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Encoder blocks
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # ASPP
        self.aspp = ASPP(512, [6, 12, 18])
        
        # Decoder
        self.decoder_conv1 = nn.Conv2d(256 + 64, 256, 3, padding=1, bias=False)
        self.decoder_bn1 = nn.BatchNorm2d(256)
        self.decoder_conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.decoder_bn2 = nn.BatchNorm2d(256)
        self.decoder_conv3 = nn.Conv2d(256, out_channels, 1)
        
        # Low-level features conversion
        self.low_level_conv = nn.Conv2d(64, 64, 1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(64)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Save low-level features
        low_level_feat = x
        
        # Continue encoding
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        # Process low-level features
        low_level_feat = self.low_level_conv(low_level_feat)
        low_level_feat = self.low_level_bn(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
        
        # Upsample ASPP features
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate with low-level features
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Final convolutions
        x = self.decoder_conv1(x)
        x = self.decoder_bn1(x)
        x = self.relu(x)
        x = self.decoder_conv2(x)
        x = self.decoder_bn2(x)
        x = self.relu(x)
        
        # Upsample to original size
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.decoder_conv3(x)
        
        return x