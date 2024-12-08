'''


단일 라벨 세그멘테이션에 효과적인 모델들을 소개해드리겠습니다:

### 1. U-Net 계열

```python
# U-Net++ 구현 예시
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Nested Decoder
        self.dec1_0 = self.conv_block(512 + 256, 256)
        self.dec2_0 = self.conv_block(256 + 128, 128)
        self.dec3_0 = self.conv_block(128 + 64, 64)
        
        # 최종 출력층
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Dense skip connections 구현
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        dec1 = self.dec1_0(torch.cat([enc3, enc4], dim=1))
        dec2 = self.dec2_0(torch.cat([enc2, dec1], dim=1))
        dec3 = self.dec3_0(torch.cat([enc1, dec2], dim=1))
        
        return self.final(dec3)
```

**장점:**
- 의료 영상 등 작은 객체 검출에 강점
- 적은 데이터셋으로도 좋은 성능
- Skip connection으로 세부 정보 보존

### 2. DeepLab V3+

```python
class DeepLabV3Plus(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16):
        super().__init__()
        
        # ASPP (Atrous Spatial Pyramid Pooling)
        self.aspp = ASPP(
            in_channels=2048,
            out_channels=256,
            dilations=[1, 6, 12, 18]
        )
        
        # Decoder
        self.decoder = Decoder(
            low_level_channels=256,
            num_classes=1
        )
    
    def forward(self, x):
        # Backbone features
        low_level_feat, encoder_output = self.backbone(x)
        
        # ASPP
        aspp_output = self.aspp(encoder_output)
        
        # Decoder
        output = self.decoder(aspp_output, low_level_feat)
        
        return output
```

**장점:**
- 다양한 스케일의 객체 검출 가능
- 경계 검출이 정확함
- 고해상도 특징 맵 유지

### 3. HRNet (High-Resolution Network)

```python
class HRNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 다중 해상도 스트림
        self.stream1 = self.make_stream(channels=64)  # 고해상도
        self.stream2 = self.make_stream(channels=128) # 중간해상도
        self.stream3 = self.make_stream(channels=256) # 저해상도
        
        # 융합 모듈
        self.fusion = MultiResolutionFusion()
        
        # 출력 헤드
        self.head = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        # 병렬 처리
        s1 = self.stream1(x)
        s2 = self.stream2(x)
        s3 = self.stream3(x)
        
        # 특징 융합
        fused = self.fusion([s1, s2, s3])
        
        return self.head(fused)
```

**장점:**
- 고해상도 특징 유지
- 정교한 경계 검출
- 다양한 크기의 객체 검출

### 4. 최적화된 학습 파이프라인

```python
class SegmentationTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler()  # Mixed precision
        
    def train_step(self, images, masks):
        with autocast():
            # 예측
            pred_masks = self.model(images)
            
            # 손실 계산 (Dice Loss + BCE)
            loss = self.criterion(pred_masks, masks)
            
            # 역전파
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
        return loss.item()
    
    def validate(self, val_loader):
        self.model.eval()
        dice_scores = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                pred_masks = self.model(images)
                dice_score = calculate_dice(pred_masks, masks)
                dice_scores.append(dice_score)
                
        return np.mean(dice_scores)
```

### 5. 데이터 증강 전략

```python
def get_augmentation():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ])
```

### 모델 선택 기준:

1. **작은 객체 검출이 중요한 경우:**
   - U-Net++ 또는 U-Net 3+
   - 고해상도 특징 맵 유지
   - Dense skip connections

2. **경계가 복잡한 경우:**
   - DeepLab V3+
   - HRNet
   - 다중 스케일 처리 능력

3. **실시간 처리가 필요한 경우:**
   - EfficientUNet
   - MobileNetV3 + U-Net
   - 경량화된 모델 구조

4. **메모리 제약이 있는 경우:**
   - U-Net with MobileNet backbone
   - Lightweight DeepLab
   - 파라미터 수가 적은 모델

이러한 모델들은 각각의 장단점이 있으므로, 실제 문제의 특성과 제약 조건을 고려하여 선택하시면 됩니다.
'''
'''
1024x1024 입력 이미지에 맞게 수정된 UNet 3+ 코드입니다:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_scale=4):
        super(UNet3Plus, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]
        
        ## -------------Encoder--------------
        self.conv1 = ConvBlock(in_channels, filters[0])  # 1024x1024
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = ConvBlock(filters[0], filters[1])  # 512x512
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = ConvBlock(filters[1], filters[2])  # 256x256
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = ConvBlock(filters[2], filters[3])  # 128x128
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = ConvBlock(filters[3], filters[4])  # 64x64

        ## -------------Decoder--------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        ```stage 4d```
        # h1->1024*1024, hd4->128*128, Pooling 8 times
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->512*512, hd4->128*128, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->256*256, hd4->128*128, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->128*128, hd4->128*128, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->64*64, hd4->128*128, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        ```stage 3d```
        # h1->1024*1024, hd3->256*256, Pooling 4 times
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->512*512, hd3->256*256, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->256*256, hd3->256*256, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->128*128, hd4->256*256, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->64*64, hd4->256*256, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, 3, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # Final output
        self.outconv1 = nn.Conv2d(self.UpChannels, out_channels, 3, padding=1)
        
        # Final upsampling to 1024x1024
        self.final_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs):
        ## -------------Encoder-------------
        h1 = self.conv1(inputs)  # h1->1024*1024*64

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)  # h2->512*512*128

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)  # h3->256*256*256

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)  # h4->128*128*512

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)  # h5->64*64*1024

        ## -------------Decoder-------------
        # Stage 4d
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(
            torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        # Stage 3d
        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(
            torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        # Final output
        d1 = self.outconv1(hd3)  # 256x256
        d1 = self.final_up(d1)   # Upsampling to 1024x1024
        return d1

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# 모델 사용 예시
if __name__ == '__main__':
    # 모델 초기화
    model = UNet3Plus(in_channels=3, out_channels=1)
    
    # 1024x1024 입력 데이터 생성
    x = torch.randn(1, 3, 1024, 1024)
    
    # 모델 실행
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
```

주요 변경사항:
1. 입력 이미지 크기를 1024x1024로 조정
2. 각 레이어의 feature map 크기 조정
3. 최종 출력을 1024x1024로 upsampling하는 layer 추가
4. Pooling 및 upsampling 비율 조정

메모리 사용량 최적화 팁:
1. `torch.cuda.empty_cache()`를 주기적으로 호출
2. Gradient accumulation 사용
3. Mixed precision training 적용
4. 배치 크기 조절

이 모델은 1024x1024 입력 이미지에 대해 동일한 크기의 세그멘테이션 마스크를 출력합니다.'''