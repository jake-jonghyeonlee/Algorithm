import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=True):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attention = CBAM(out_channels) if use_attention else None
        
    def forward(self, x):
        x = self.double_conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x

class PatternAwareUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):
        super().__init__()
        
        # Encoder with pattern-aware features
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)
        
        # Multi-scale feature fusion
        self.msff4 = nn.Conv2d(1024+512, 512, 1)
        self.msff3 = nn.Conv2d(512+256, 256, 1)
        self.msff2 = nn.Conv2d(256+128, 128, 1)
        self.msff1 = nn.Conv2d(128+64, 64, 1)
        
        # Decoder with attention
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        
        # 클래스별 패턴 인식을 위한 병렬 컨볼루션
        self.pattern_convs = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            for _ in range(num_classes)
        ])
        
        # 최종 출력층
        self.final_conv = nn.Conv2d(64 * num_classes, num_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        # Encoder path with residual connections
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        x = self.dropout(x)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        x = self.dropout(x)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        x = self.dropout(x)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        x = self.dropout(x)
        
        x = self.enc5(x)
        
        # Decoder path with multi-scale feature fusion
        x = self.up(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.msff4(x)
        x = self.dec4(x)
        x = self.dropout(x)
        
        x = self.up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.msff3(x)
        x = self.dec3(x)
        x = self.dropout(x)
        
        x = self.up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.msff2(x)
        x = self.dec2(x)
        x = self.dropout(x)
        
        x = self.up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.msff1(x)
        x = self.dec1(x)
        
        # 클래스별 패턴 인식
        pattern_outputs = []
        for conv in self.pattern_convs:
            pattern_outputs.append(conv(x))
        
        # 패턴 특징 결합
        x = torch.cat(pattern_outputs, dim=1)
        return self.final_conv(x)
