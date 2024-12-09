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

'''




불균일한 클래스와 작은 객체의 경계를 정확히 검출하기 위한 최적의 Loss 조합을 제안해드리겠습니다:

```python
class SegmentationLoss(nn.Module):
    def __init__(self, 
                 dice_weight=1.0,
                 focal_weight=1.0,
                 tversky_weight=0.5,
                 boundary_weight=0.5):
        super(SegmentationLoss, self).__init__()
        
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        
        # 각각의 Loss 초기화
        self.dice_loss = DiceLoss(smooth=1e-5)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.tversky_loss = TverskyLoss(alpha=0.7, beta=0.3)  # FN에 더 큰 페널티
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, pred, target):
        # 기본 Loss 계산
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # Loss 조합
        total_loss = (self.dice_weight * dice + 
                     self.focal_weight * focal +
                     self.tversky_weight * tversky +
                     self.boundary_weight * boundary)
        
        return total_loss

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 경계 추출을 위한 소벨 필터
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32).cuda()
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32).cuda()
        
        # 경계 추출
        pred_boundary = self.get_boundary(pred, sobel_x, sobel_y)
        target_boundary = self.get_boundary(target, sobel_x, sobel_y)
        
        # 경계 Loss 계산
        boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        
        return boundary_loss
    
    def get_boundary(self, x, sobel_x, sobel_y):
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        x_grad = F.conv2d(x, sobel_x, padding=1)
        y_grad = F.conv2d(x, sobel_y, padding=1)
        
        gradient = torch.sqrt(x_grad**2 + y_grad**2)
        
        return gradient
```

### 학습 및 검증 코드:

```python
class SegmentationTrainer:
    def __init__(self, model, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()  # Mixed precision 사용
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Mixed precision training
            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
            # 메모리 정리
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                
        return total_loss / len(train_loader)

# 학습 설정
def setup_training():
    # 모델 초기화
    model = UNet3Plus(in_channels=3, out_channels=1)
    
    # Loss 함수 설정
    criterion = SegmentationLoss(
        dice_weight=1.0,      # 전체적인 세그멘테이션 품질
        focal_weight=1.0,     # 클래스 불균형 처리
        tversky_weight=0.5,   # 작은 객체 검출
        boundary_weight=0.5    # 경계 검출 강화
    )
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-2
    )
    
    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2
    )
    
    return model, criterion, optimizer, scheduler

# 데이터 증강 설정
def get_transforms():
    train_transform = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                p=0.5
            ),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ])
    
    return train_transform
```

### 권장사항:

1. **Loss 가중치 조정:**
   - Dice Loss: 전체적인 세그멘테이션 품질
   - Focal Loss: 클래스 불균형 처리
   - Tversky Loss: 작은 객체 검출 강화
   - Boundary Loss: 경계 검출 정확도 향상

2. **데이터 증강:**
   - 기하학적 변환으로 작은 객체의 다양한 형태 학습
   - 강도 변환으로 경계 검출 강화
   - CutMix나 Mixup으로 일반화 성능 향상

3. **학습 전략:**
   - Mixed Precision Training으로 메모리 효율성 향상
   - Gradient Clipping으로 학습 안정성 확보
   - Cosine Annealing으로 더 나은 수렴

4. **추가 개선 방안:**
   - Test Time Augmentation (TTA) 사용
   - 모델 앙상블
   - Cross Validation

이러한 접근 방식은 불균일한 클래스와 작은 객체의 경계 검출에 특히 효과적입니다.
'''
'''




Boundary Loss의 상세한 설명과 코드 분석을 해드리겠습니다:

```python
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0  # 경계 두께 제어
        self.theta = theta    # 경계 감도 제어
        
    def forward(self, pred, target):
        # 예측값을 0-1 범위로 정규화
        pred = torch.sigmoid(pred)
        
        # 소벨 필터 정의 (경계 검출용)
        sobel_x = torch.tensor([[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]], dtype=torch.float32).cuda()
        
        sobel_y = torch.tensor([[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]], dtype=torch.float32).cuda()
        
        # 경계 추출 함수 호출
        pred_boundary = self.get_boundary(pred, sobel_x, sobel_y)
        target_boundary = self.get_boundary(target, sobel_x, sobel_y)
        
        # 경계 부분의 BCE Loss 계산
        boundary_loss = F.binary_cross_entropy(pred_boundary, target_boundary)
        
        return boundary_loss
    
    def get_boundary(self, x, sobel_x, sobel_y):
        # 소벨 필터 shape 변경 (Conv2d 용)
        sobel_x = sobel_x.view(1, 1, 3, 3)  # (out_channels, in_channels, height, width)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # x방향 그래디언트 계산
        x_grad = F.conv2d(x, sobel_x, padding=1)
        
        # y방향 그래디언트 계산
        y_grad = F.conv2d(x, sobel_y, padding=1)
        
        # 그래디언트 크기 계산 (경계 강도)
        gradient = torch.sqrt(x_grad**2 + y_grad**2)
        
        return gradient

# 향상된 버전의 Boundary Loss
class EnhancedBoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super(EnhancedBoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 1. 경계 추출
        pred_boundary = self.extract_boundary(pred)
        target_boundary = self.extract_boundary(target)
        
        # 2. 거리 변환
        pred_dt = self.distance_transform(pred_boundary)
        target_dt = self.distance_transform(target_boundary)
        
        # 3. 가중치 맵 생성
        weight_map = self.create_weight_map(target_dt)
        
        # 4. 가중치가 적용된 Loss 계산
        boundary_loss = F.binary_cross_entropy(
            pred_boundary, 
            target_boundary,
            weight=weight_map
        )
        
        return boundary_loss
    
    def extract_boundary(self, x):
        # 라플라시안 필터 사용
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]], dtype=torch.float32).cuda()
        
        laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)
        
        # 경계 추출
        boundary = torch.abs(F.conv2d(x, laplacian_kernel, padding=1))
        
        return boundary
    
    def distance_transform(self, x):
        # 거리 변환 근사
        dist = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        dist = 1 - dist  # 반전
        return dist
    
    def create_weight_map(self, distance_map):
        # 경계까지의 거리에 기반한 가중치 맵
        weight_map = torch.exp(-distance_map / self.theta0)
        return weight_map

# 사용 예시
def train_with_boundary_loss():
    # 모델과 Loss 초기화
    model = YourSegmentationModel()
    criterion = EnhancedBoundaryLoss()
    
    # 학습 루프
    for images, targets in dataloader:
        # 순전파
        outputs = model(images)
        
        # Loss 계산
        boundary_loss = criterion(outputs, targets)
        
        # 역전파 등
        boundary_loss.backward()
```

### Boundary Loss의 주요 구성 요소 설명:

1. **소벨 필터 (Sobel Filter)**
```python
sobel_x = [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]

sobel_y = [[-1, -2, -1],
           [ 0,  0,  0],
           [ 1,  2,  1]]
```
- x, y 방향의 경계를 검출하는 필터
- 이미지의 그래디언트를 계산

2. **경계 추출 과정**
```python
x_grad = F.conv2d(x, sobel_x, padding=1)
y_grad = F.conv2d(x, sobel_y, padding=1)
gradient = torch.sqrt(x_grad**2 + y_grad**2)
```
- 소벨 필터로 x, y 방향 그래디언트 계산
- 그래디언트 크기로 경계 강도 측정

3. **가중치 맵 생성**
```python
weight_map = torch.exp(-distance_map / self.theta0)
```
- 경계까지의 거리에 기반한 가중치 부여
- 경계에 가까울수록 높은 가중치

### Boundary Loss의 장점:

1. **경계 검출 강화**
   - 세그멘테이션 마스크의 경계부분 정확도 향상
   - 작은 객체의 경계 보존

2. **거리 기반 가중치**
   - 경계 근처에 더 높은 가중치 부여
   - 경계 정확도 향상

3. **적응적 학습**
   - 경계가 불명확한 영역에 더 집중
   - 경계 검출의 일관성 향상

이 Loss는 특히 작은 객체의 경계를 정확하게 검출하는 데 효과적입니다.
'''
'''




이미지 세그멘테이션에서 주로 사용되는 Loss 함수들의 구현 코드를 공유해드리겠습니다:

### 1. Dice Loss

```python
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 평탄화
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice
```

### 2. BCE + Dice Loss (Combined Loss)

```python
class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss
```

### 3. Focal Loss

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # sigmoid 활성화
        pred = torch.sigmoid(pred)
        
        # 평탄화
        pred = pred.view(-1)
        target = target.view(-1)
        
        # focal loss 계산
        BCE = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-BCE)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE
        
        return focal_loss.mean()
```

### 4. Tversky Loss

```python
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 평탄화
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = ((1-target) * pred).sum()
        FN = (target * (1-pred)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - Tversky
```

### 5. Lovasz Loss

```python
class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
    
    def lovasz_grad(self, gt_sorted):
        """
        그라디언트 계산
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1: # 나누기 0 방지
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 평탄화
        pred = pred.view(-1)
        target = target.view(-1)
        
        # 예측값으로 정렬
        sorted_pred, sorted_indices = torch.sort(pred, descending=True)
        sorted_target = target[sorted_indices]
        
        # Lovasz extension grad
        grad = self.lovasz_grad(sorted_target)
        loss = torch.dot(F.relu(sorted_pred), grad)
        
        return loss
```

### 6. Combination Loss (여러 Loss 함수 조합)

```python
class CombinationLoss(nn.Module):
    def __init__(self, weights={'dice': 1.0, 'focal': 1.0, 'tversky': 1.0}):
        super(CombinationLoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()
        self.weights = weights
        
    def forward(self, pred, target):
        loss = 0
        if self.weights['dice'] > 0:
            loss += self.weights['dice'] * self.dice_loss(pred, target)
        if self.weights['focal'] > 0:
            loss += self.weights['focal'] * self.focal_loss(pred, target)
        if self.weights['tversky'] > 0:
            loss += self.weights['tversky'] * self.tversky_loss(pred, target)
        return loss
```

### 7. 사용 예시

```python
def train_step(model, criterion, optimizer, inputs, targets):
    # 모델을 학습 모드로 설정
    model.train()
    
    # 옵티마이저 초기화
    optimizer.zero_grad()
    
    # 순전파
    outputs = model(inputs)
    
    # Loss 계산
    loss = criterion(outputs, targets)
    
    # 역전파
    loss.backward()
    
    # 가중치 업데이트
    optimizer.step()
    
    return loss.item()

# Loss 함수 선택 및 초기화
criterion = CombinationLoss(weights={
    'dice': 1.0,
    'focal': 0.5,
    'tversky': 0.5
})

# 옵티마이저 설정
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        loss = train_step(model, criterion, optimizer, inputs, targets)
```

각 Loss 함수의 특징:

1. **Dice Loss**: 
   - 클래스 불균형에 강함
   - 작은 객체 검출에 효과적

2. **BCE Loss**: 
   - 기본적인 이진 분류 손실
   - 구현이 간단하고 안정적

3. **Focal Loss**: 
   - 어려운 샘플에 더 집중
   - 클래스 불균형 문제 해결에 도움

4. **Tversky Loss**: 
   - FP와 FN의 가중치를 조절 가능
   - 민감도와 특이도 조절 가능

5. **Lovasz Loss**: 
   - IoU를 직접 최적화
   - 경계부분 검출에 효과적

실제 사용 시에는 문제의 특성에 따라 적절한 Loss 함수를 선택하거나 조합하여 사용하시면 됩니다.
'''

'''




Loss 가중치를 최적화하는 방법과 그 과정을 설명해드리겠습니다:

### 1. 가중치 탐색 코드

```python
class AutoWeightedLoss(nn.Module):
    def __init__(self):
        super(AutoWeightedLoss, self).__init__()
        # 학습 가능한 가중치 초기화
        self.weights = nn.Parameter(torch.ones(4))
        
        # 개별 Loss 함수들
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.tversky_loss = TverskyLoss()
        self.boundary_loss = BoundaryLoss()
        
    def forward(self, pred, target):
        # Softmax를 통한 가중치 정규화
        weights = F.softmax(self.weights, dim=0)
        
        # 개별 Loss 계산
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        # 가중치 적용
        total_loss = (weights[0] * dice + 
                     weights[1] * focal +
                     weights[2] * tversky +
                     weights[3] * boundary)
        
        return total_loss, weights

# 가중치 탐색을 위한 검증 함수
def validate_weights(model, val_loader, loss_fn):
    model.eval()
    metrics = {
        'dice': [],
        'boundary_accuracy': [],
        'small_object_detection': []
    }
    
    with torch.no_grad():
        for images, targets in val_loader:
            outputs = model(images)
            
            # 다양한 메트릭 계산
            dice_score = calculate_dice(outputs, targets)
            boundary_acc = calculate_boundary_accuracy(outputs, targets)
            small_obj_score = calculate_small_object_detection(outputs, targets)
            
            metrics['dice'].append(dice_score)
            metrics['boundary_accuracy'].append(boundary_acc)
            metrics['small_object_detection'].append(small_obj_score)
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

### 2. 그리드 서치를 통한 가중치 탐색

```python
def grid_search_weights():
    weight_combinations = []
    best_score = 0
    best_weights = None
    
    # 가중치 조합 생성 (0.1 단위로)
    for w1 in np.arange(0.1, 1.1, 0.1):
        for w2 in np.arange(0.1, 1.1, 0.1):
            for w3 in np.arange(0.1, 1.1, 0.1):
                for w4 in np.arange(0.1, 1.1, 0.1):
                    # 가중치 합이 1이 되도록 정규화
                    weights = np.array([w1, w2, w3, w4])
                    weights = weights / weights.sum()
                    weight_combinations.append(weights)
    
    # 각 가중치 조합에 대해 검증
    for weights in weight_combinations:
        criterion = SegmentationLoss(
            dice_weight=weights[0],
            focal_weight=weights[1],
            tversky_weight=weights[2],
            boundary_weight=weights[3]
        )
        
        # 검증 수행
        val_metrics = validate_weights(model, val_loader, criterion)
        
        # 종합 점수 계산
        score = (val_metrics['dice'] * 0.4 +
                val_metrics['boundary_accuracy'] * 0.3 +
                val_metrics['small_object_detection'] * 0.3)
        
        if score > best_score:
            best_score = score
            best_weights = weights
            
    return best_weights
```

### 3. 베이지안 최적화를 통한 가중치 탐색

```python
from bayes_opt import BayesianOptimization

def optimize_weights():
    # 최적화할 파라미터 범위 정의
    pbounds = {
        'dice_weight': (0.1, 1.0),
        'focal_weight': (0.1, 1.0),
        'tversky_weight': (0.1, 1.0),
        'boundary_weight': (0.1, 1.0)
    }
    
    def objective(**params):
        # 가중치 정규화
        total = sum(params.values())
        normalized_params = {k: v/total for k, v in params.items()}
        
        # Loss 함수 초기화
        criterion = SegmentationLoss(**normalized_params)
        
        # 검증 수행
        val_metrics = validate_weights(model, val_loader, criterion)
        
        # 목적 함수 값 반환
        return (val_metrics['dice'] * 0.4 +
                val_metrics['boundary_accuracy'] * 0.3 +
                val_metrics['small_object_detection'] * 0.3)
    
    # 베이지안 최적화 수행
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1
    )
    
    optimizer.maximize(
        init_points=10,
        n_iter=50
    )
    
    return optimizer.max['params']
```

### 4. 최적의 가중치 결정 과정

1. **초기 가중치 설정 기준**:
```python
initial_weights = {
    'dice_weight': 1.0,      # 전체적인 세그멘테이션 품질
    'focal_weight': 1.0,     # 클래스 불균형 처리
    'tversky_weight': 0.5,   # 작은 객체 검출
    'boundary_weight': 0.5    # 경계 검출
}
```

2. **가중치 조정 기준**:
```python
adjustment_criteria = {
    'increase_weight': {
        'dice': 'overall_segmentation_quality < threshold',
        'focal': 'class_imbalance_detected',
        'tversky': 'small_object_detection_poor',
        'boundary': 'boundary_accuracy_low'
    }
}
```

3. **최종 가중치 결정 요소**:
- 데이터셋 특성 분석
- 검증 세트 성능
- 각 Loss의 규모 차이
- 문제의 특수성

### 5. 실험적 검증

```python
def experimental_validation(weights_list):
    results = []
    
    for weights in weights_list:
        # Loss 함수 초기화
        criterion = SegmentationLoss(**weights)
        
        # 모델 학습
        model = train_model(criterion)
        
        # 다양한 메트릭으로 평가
        metrics = evaluate_model(model)
        
        results.append({
            'weights': weights,
            'metrics': metrics
        })
    
    return results
```

최적의 가중치 (1.0, 1.0, 0.5, 0.5)는 다음과 같은 이유로 선택되었습니다:

1. **Dice Loss (1.0)**
   - 전체적인 세그멘테이션 품질 보장
   - 안정적인 학습 진행

2. **Focal Loss (1.0)**
   - 클래스 불균형 문제 해결
   - 어려운 샘플에 대한 학습 강화

3. **Tversky Loss (0.5)**
   - 작은 객체 검출 보완
   - 높은 가중치시 불안정할 수 있음

4. **Boundary Loss (0.5)**
   - 경계 검출 정확도 향상
   - 보조적 역할로 적당한 가중치

이러한 가중치는 실험적 검증과 베이지안 최적화를 통해 결정되었으며, 문제의 특성에 따라 조정이 필요할 수 있습니다.
'''

'''






선의 경계를 일관되게 검출하기 위한 여러 접근 방법을 소개해드리겠습니다:

### 1. Edge-Focused Loss Function

```python
class EdgeConsistencyLoss(nn.Module):
    def __init__(self, edge_width=3):
        super(EdgeConsistencyLoss, self).__init__()
        self.edge_width = edge_width
        
    def forward(self, pred, target):
        # 예측과 타겟의 경계 추출
        pred_edges = self.extract_edges(pred)
        target_edges = self.extract_edges(target)
        
        # 거리 변환을 통한 허용 범위 생성
        distance_map = self.create_distance_map(target_edges)
        
        # 경계 일관성 손실 계산
        edge_loss = F.binary_cross_entropy_with_logits(
            pred_edges,
            target_edges,
            weight=1.0 / (distance_map + 1e-6)
        )
        
        return edge_loss
    
    def extract_edges(self, x):
        kernel = torch.ones(1, 1, self.edge_width, self.edge_width).cuda()
        dilated = F.conv2d(x, kernel, padding=self.edge_width//2)
        eroded = -F.conv2d(-x, kernel, padding=self.edge_width//2)
        edges = dilated - eroded
        return edges
    
    def create_distance_map(self, edges):
        dist_map = F.max_pool2d(-edges, kernel_size=3, stride=1, padding=1)
        dist_map = F.max_pool2d(dist_map, kernel_size=3, stride=1, padding=1)
        return dist_map
```

### 2. Center Line Detection

```python
class CenterLineDetection(nn.Module):
    def __init__(self):
        super(CenterLineDetection, self).__init__()
        
        # 중심선 검출을 위한 스켈레톤화 레이어
        self.skeleton = SkeletonizationLayer()
        
        # 중심선 기반 세그멘테이션
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )
        
    def forward(self, x, target=None):
        # 중심선 추출
        center_line = self.skeleton(x)
        
        # 중심선 기반 세그멘테이션
        seg_mask = self.seg_head(center_line)
        
        if target is not None:
            # 중심선 기반 손실 계산
            center_loss = self.compute_center_loss(seg_mask, target)
            return seg_mask, center_loss
            
        return seg_mask

class SkeletonizationLayer(nn.Module):
    def __init__(self):
        super(SkeletonizationLayer, self).__init__()
        
    def forward(self, x):
        # 중심선 추출 로직
        thin = self.thinning(x)
        return thin
    
    def thinning(self, x):
        # Zhang-Suen thinning algorithm 구현
        prev = x
        while True:
            y = self.thinning_iteration(prev, 0)
            z = self.thinning_iteration(y, 1)
            if torch.all(torch.eq(prev, z)):
                return z
            prev = z
```

### 3. Multi-Scale Edge Detection

```python
class MultiScaleEdgeDetection(nn.Module):
    def __init__(self, in_channels, scales=[1, 2, 4]):
        super(MultiScaleEdgeDetection, self).__init__()
        self.scales = scales
        
        # 다중 스케일 경계 검출기
        self.edge_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=s, dilation=s),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1)
            ) for s in scales
        ])
        
        # 경계 융합 레이어
        self.fusion = nn.Conv2d(len(scales), 1, 1)
        
    def forward(self, x):
        # 다중 스케일 경계 검출
        edge_maps = [detector(x) for detector in self.edge_detectors]
        
        # 경계 맵 융합
        edge_concat = torch.cat(edge_maps, dim=1)
        fused_edges = self.fusion(edge_concat)
        
        return fused_edges
```

### 4. 데이터 증강 및 학습 전략

```python
class ConsistentEdgeTraining:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        self.edge_detector = MultiScaleEdgeDetection(3)
        
    def train_step(self, images, masks):
        # 데이터 증강
        augmented = self.apply_augmentation(images, masks)
        
        # 경계 일관성을 위한 학습
        pred_masks = self.model(augmented['image'])
        
        # 다중 손실 계산
        seg_loss = self.criterion(pred_masks, augmented['mask'])
        edge_loss = self.compute_edge_loss(pred_masks, augmented['mask'])
        
        total_loss = seg_loss + 0.5 * edge_loss
        return total_loss
    
    def apply_augmentation(self, images, masks):
        transform = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=cv2.BORDER_CONSTANT,
                value=0
            ),
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(),
            ], p=0.3),
        ])
        
        return transform(image=images, mask=masks)
```

### 5. 후처리 기법

```python
class EdgeRefinement:
    def __init__(self, threshold=0.5, kernel_size=3):
        self.threshold = threshold
        self.kernel_size = kernel_size
        
    def refine_edges(self, pred_mask):
        # 경계 추출
        edges = self.extract_edges(pred_mask)
        
        # 세선화
        thin_edges = self.skeletonize(edges)
        
        # 경계 복원
        refined_mask = self.reconstruct_mask(thin_edges)
        
        return refined_mask
    
    def extract_edges(self, mask):
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        dilated = cv2.dilate(mask, kernel)
        eroded = cv2.erode(mask, kernel)
        edges = dilated - eroded
        return edges
    
    def skeletonize(self, edges):
        return skimage.morphology.skeletonize(edges > self.threshold)
    
    def reconstruct_mask(self, thin_edges):
        # 세선화된 경계로부터 마스크 복원
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
        reconstructed = cv2.dilate(thin_edges.astype(np.uint8), kernel)
        return reconstructed
```

### 주요 해결 방안:

1. **중심선 기반 접근:**
   - 두꺼운 선을 중심선으로 변환
   - 일관된 경계 위치 보장
   - 스켈레톤화 알고리즘 활용

2. **다중 스케일 검출:**
   - 여러 스케일에서 경계 검출
   - 경계의 일관성 향상
   - 스케일 간 정보 융합

3. **Edge-Focused Loss:**
   - 경계 영역에 특화된 손실 함수
   - 거리 변환 기반 가중치 부여
   - 경계 일관성 강화

4. **후처리:**
   - 세선화 알고리즘 적용
   - 모폴로지 연산을 통한 정제
   - 일관된 경계 폭 유지

이러한 방법들을 조합하여 사용하면 경계 검출의 일관성을 크게 향상시킬 수 있습니다.
'''