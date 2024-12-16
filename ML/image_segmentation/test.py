import torch
import cv2
import numpy as np
from pathlib import Path
from models import UNet, AttentionUNet, ResUNet, DenseUNet, SEUNet, DeepLabV3Plus
from augmentation import get_transforms
import matplotlib.pyplot as plt

def load_model(checkpoint_path):
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path)
    
    # 모델 타입에 따른 모델 생성
    model_type = checkpoint['model_type']
    if model_type == "UNet":
        model = UNet()
    elif model_type == "AttentionUNet":
        model = AttentionUNet()
    elif model_type == "ResUNet":
        model = ResUNet()
    elif model_type == "DenseUNet":
        model = DenseUNet()
    elif model_type == "DeepLabV3Plus":
        model = DeepLabV3Plus()
    else:
        model = SEUNet()
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['transform_params']

def predict_mask(model, image, transform):
    # 이미지 전처리
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    
    # Transform 적용
    augmented = transform(image=image)
    image = augmented['image']
    
    # 배치 차원 추가
    image = image.unsqueeze(0)
    
    # 예측
    model.eval()
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    # 배치 차원 제거 및 CPU로 이동
    pred = pred.squeeze().cpu().numpy()
    return pred

def visualize_results(image, mask, save_path):
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 예측 마스크
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # 오버레이
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    overlay[mask == 1] = [255, 0, 0]  # 빨간색으로 마스크 표시
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 경로 설정
    model_path = Path("models/best_model.pkl")
    test_dir = Path("data/images/test")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform_params = load_model(model_path)
    model = model.to(device)
    
    # Transform 설정
    transform = get_transforms(train=False, aug_params=transform_params)
    
    # 테스트 이미지 처리
    for img_path in test_dir.glob('*.png'):  # 또는 *.jpg
        # 이미지 로드
        image = cv2.imread(str(img_path))
        
        # 마스크 예측
        mask = predict_mask(model, image, transform)
        
        # 결과 저장
        output_path = output_dir / f"{img_path.stem}_result.png"
        visualize_results(image, mask, output_path)
        
        # 마스크 파일 따로 저장
        mask_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        
        print(f"Processed: {img_path.name}")

if __name__ == "__main__":
    main() 

    '''
    data/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── masks/
    ├── train/
    └── valid/

models/
└── best_model.pkl

results/
├── image1_result.png
├── image1_mask.png
├── image2_result.png
└── image2_mask.png'''

'''
3. **메모리 효율적인 skip connection 구현**:
class UNet(nn.Module):
    def forward(self, x):
        # 인코더 특징을 리스트에 저장하지 않고 바로 사용
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # 필요할 때만 특징을 사용
        dec3 = self.up3(enc4)
        del enc4  # 더 이상 필요없는 특징 삭제
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        del enc3
        
        dec2 = self.up2(dec3)
        del dec3
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        del enc2
        
        dec1 = self.up1(dec2)
        del dec2
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        del enc1
        
        return self.final(dec1)

4. **torch.cuda.empty_cache() 사용**:
import torch
import gc

def train_epoch():
    for batch in dataloader:
        # 학습 코드
        pass
    torch.cuda.empty_cache()
    gc.collect()



lr_scheduler를 사용하는 주요 이유들을 설명드리겠습니다:

1. **학습률 동적 조정**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=5,
    factor=0.1
)

# 학습 중 사용
for epoch in range(epochs):
    train_loss = train_epoch()
    scheduler.step(train_loss)  # 검증 손실에 따라 학습률 조정
```

2. **Local Minima 회피**
```python
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,
    max_lr=1e-3,
    cycle_momentum=False
)
```

3. **수렴 안정성 향상**
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,  # 30 에포크마다
    gamma=0.1      # 학습률을 1/10로 감소
)
```

4. **초기 빠른 학습과 후기 미세 조정**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)
```

주요 장점들:

1. **학습 안정성 향상**
   - 초기에는 큰 학습률로 빠르게 학습
   - 후반에는 작은 학습률로 미세 조정

2. **과적합 방지**
   - 적절한 학습률 감소로 모델의 일반화 성능 향상

3. **학습 시간 단축**
   - 효율적인 학습률 조정으로 빠른 수렴 가능

4. **성능 개선**
   - 더 좋은 최적해 도달 가능

자주 사용되는 스케줄러 예시:

```python
# 1. ReduceLROnPlateau - 성능 개선이 없을 때 학습률 감소
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',
    patience=5,
    factor=0.1
)

# 2. StepLR - 정해진 에포크마다 학습률 감소
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)

# 3. CosineAnnealingLR - 코사인 함수처럼 학습률 주기적 변화
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=100,  # 주기
    eta_min=1e-6  # 최소 학습률
)

# 4. OneCycleLR - 한 사이클 정책으로 학습률 조정
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=epochs,
    steps_per_epoch=len(train_loader)
)
```

사용 시 주의사항:

1. **적절한 스케줄러 선택**
   - 문제와 데이터셋에 맞는 스케줄러 선택 필요

2. **하이퍼파라미터 튜닝**
   - patience, factor 등의 파라미터 적절히 조정

3. **모니터링**
```python
current_lr = optimizer.param_groups[0]['lr']
print(f"Current learning rate: {current_lr}")
```

이러한 학습률 스케줄링은 딥러닝 모델의 학습 과정을 최적화하는 중요한 기법입니다.

        '''