import cv2
import numpy as np
from PIL import Image

def preprocess_image_for_segmentation(image_path, target_size=(256, 256), apply_clahe=True):
    """
    이미지 세그멘테이션을 위한 이미지 전처리 함수
    
    Args:
        image_path: 이미지 파일 경로
        target_size: 조정할 이미지 크기 (width, height)
        apply_clahe: CLAHE 히스토그램 평활화 적용 여부
    
    Returns:
        preprocessed_image: 전처리된 이미지
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 이미지 크기 조정
    img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    if apply_clahe:
        # CLAHE 히스토그램 평활화 적용
        img_resized = apply_histogram_equalization(img_resized)
    
    # 이미지 정규화 (0-1 범위로)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # 필요한 경우 채널 차원 추가 (배치 처리를 위해)
    img_expanded = np.expand_dims(img_normalized, axis=0)
    
    return img_expanded

def apply_histogram_equalization(image):
    """
    CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용 함수
    
    Args:
        image: 입력 이미지
    
    Returns:
        equalized_image: 히스토그램 평활화가 적용된 이미지
    """
    # RGB를 LAB 색공간으로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE 객체 생성
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # L 채널에 대해 CLAHE 적용
    l_equalized = clahe.apply(l)
    
    # 채널 합치기
    lab_equalized = cv2.merge([l_equalized, a, b])
    
    # LAB를 RGB로 다시 변환
    equalized_image = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2RGB)
    
    return equalized_image

def apply_augmentation(image):
    """
    이미지 증강을 위한 함수
    
    Args:
        image: 입력 이미지
    
    Returns:
        augmented_image: 증강된 이미지
    """
    # 랜덤 회전
    angle = np.random.uniform(-20, 20)
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    
    # 랜덤 밝기 조정
    brightness = np.random.uniform(0.8, 1.2)
    brightened = cv2.multiply(rotated, brightness)
    
    # 가우시안 노이즈 추가
    noise = np.random.normal(0, 0.05, image.shape)
    noisy_image = np.clip(brightened + noise, 0, 1)
    
    return noisy_image

# 사용 예시
if __name__ == "__main__":
    image_path = "sample_image.jpg"
    
    # 히스토그램 평활화가 포함된 이미지 전처리
    processed_image = preprocess_image_for_segmentation(
        image_path, 
        target_size=(256, 256), 
        apply_clahe=True
    )
    
    # 이미지 증강 적용
    augmented_image = apply_augmentation(processed_image[0])

'''
K-means clustering과 CLAHE는 목적과 동작 방식이 매우 다른 알고리즘입니다. 두 방법을 비교해보겠습니다:

### 1. 목적의 차이

**K-means Clustering:**
- 이미지의 색상을 K개의 대표 색상으로 군집화
- 이미지 단순화, 색상 양자화에 사용
- 세그멘테이션을 위한 영역 분할에 활용

**CLAHE:**
- 이미지의 대비(contrast) 향상
- 세부 특징을 더 잘 보이게 만듦
- 전처리 단계에서 특징 강화에 사용

### 2. 세그멘테이션 관점에서의 비교

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

def compare_kmeans_clahe(image_path):
    # 이미지 로드
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # CLAHE 적용
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    clahe_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # K-means 적용
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5  # 클러스터 수
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    kmeans_img = centers[labels.flatten()]
    kmeans_img = kmeans_img.reshape(img.shape)
    
    return img, clahe_img, kmeans_img
```

### 3. 효과 비교

**K-means의 장점:**
- 명확한 영역 구분 가능
- 색상 기반 세그멘테이션에 효과적
- 노이즈 감소 효과

**K-means의 단점:**
- 세부 텍스처 정보 손실
- 클러스터 수(K) 선정이 결과에 큰 영향
- 경계가 불명확한 영역에서 성능 저하

**CLAHE의 장점:**
- 세부 특징 보존
- 지역적 특성 반영
- 원본 이미지의 구조 유지

**CLAHE의 단점:**
- 직접적인 세그멘테이션은 불가능
- 노이즈도 함께 강화될 수 있음

### 4. 실제 활용 방안

**함께 사용하는 것이 효과적인 경우:**
```python
def enhanced_segmentation(image_path):
    # 1. CLAHE로 전처리
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[...,0] = clahe.apply(lab[...,0])
    enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 2. K-means로 세그멘테이션
    pixel_values = enhanced_img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 5
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    return labels.reshape((img.shape[0], img.shape[1]))
```

### 5. 권장 사용 시나리오

1. **세부 특징이 중요한 경우:**
   - CLAHE를 전처리로 사용
   - 의료 영상, 텍스처 분석 등

2. **색상 기반 분할이 필요한 경우:**
   - K-means 직접 사용
   - 간단한 객체 분할, 색상 단순화 등

3. **복잡한 세그멘테이션:**
   - CLAHE로 전처리 후 K-means 적용9ㄴㅁ 
   - 또는 다른 고급 세그멘테이션 알고리즘과 조합

결론적으로, CLAHE와 K-means는 서로 보완적인 관계에 있으며, 목적에 따라 적절히 선택하거나 조합하여 사용하는 것이 좋습니다.
'''
'''


PyTorch에서 Focal Loss와 BCE Loss의 구현과 최적화 방법을 설명해드리겠습니다.

### 1. BCE Loss 구현

```python
import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(BCELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # 수치 안정성을 위한 epsilon
        eps = 1e-7
        
        # 입력값을 0과 1 사이로 클리핑
        inputs = torch.clamp(inputs, eps, 1 - eps)
        
        # BCE Loss 계산
        bce = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        
        # 가중치가 있는 경우 적용
        if self.weight is not None:
            bce = self.weight * bce
            
        # reduction 방법 적용
        if self.reduction == 'mean':
            return torch.mean(bce)
        elif self.reduction == 'sum':
            return torch.sum(bce)
        else:
            return bce
```

### 2. Focal Loss 구현

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 수치 안정성을 위한 epsilon
        eps = 1e-7
        
        # 입력값을 0과 1 사이로 클리핑
        inputs = torch.clamp(inputs, eps, 1 - eps)
        
        # focal loss 계산
        ce_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        # reduction 방법 적용
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss
```

### 3. 손실 함수 최적화 방법

```python
# 학습 설정
def setup_training(model, learning_rate=1e-4):
    # 옵티마이저 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습률 스케줄러 설정
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    return optimizer, scheduler

# 학습 루프
def train_loop(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # GPU로 데이터 이동 (가능한 경우)
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑 (선택사항)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 가중치 업데이트
            optimizer.step()
            
            running_loss += loss.item()
        
        # 에폭당 평균 손실
        epoch_loss = running_loss / len(train_loader)
        
        # 학습률 조정
        scheduler.step(epoch_loss)
```

### 4. 최적화를 위한 추가 기법들

```python
class EnhancedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(EnhancedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # 온라인 하드 예제 마이닝 (OHEM)
        def online_hard_example_mining(loss, ratio=0.7):
            k = int(loss.numel() * ratio)
            return torch.topk(loss.view(-1), k)[0]
            
        # 기본 focal loss 계산
        focal_loss = self.compute_focal_loss(inputs, targets)
        
        # OHEM 적용
        if self.training:
            focal_loss = online_hard_example_mining(focal_loss)
            
        # reduction 적용
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss
```

### 5. 학습 최적화 팁

1. **하이퍼파라미터 최적화**
```python
# Focal Loss 하이퍼파라미터 탐색
def search_focal_params():
    param_grid = {
        'alpha': [0.25, 0.5, 0.75],
        'gamma': [1.5, 2.0, 2.5]
    }
    
    best_params = {
        'alpha': 0.25,
        'gamma': 2.0,
        'val_loss': float('inf')
    }
    
    return best_params
```

2. **데이터 증강**
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
```

3. **배치 크기 및 학습률 조정**
```python
# 점진적 배치 크기 증가
batch_sizes = [16, 32, 64, 128]
learning_rates = [1e-4, 5e-4, 1e-3]

for batch_size in batch_sizes:
    for lr in learning_rates:
        # 학습 수행
        train_loop(model, batch_size, lr)
```

이러한 구현과 최적화 방법을 통해 더 효과적인 학습이 가능합니다. 데이터셋의 특성에 따라 적절한 손실 함수와 최적화 방법을 선택하시면 됩니다.
'''