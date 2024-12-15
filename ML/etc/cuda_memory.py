'''

CUDA 메모리 에러를 해결하고 배치 크기를 효과적으로 늘릴 수 있는 여러 방법들을 소개해드리겠습니다:

### 1. Gradient Accumulation (그래디언트 누적)

```python
class Trainer:
    def __init__(self, model, optimizer, criterion, accumulation_steps=4):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accumulation_steps = accumulation_steps
        
    def train(self, train_loader):
        self.model.train()
        running_loss = 0.0
        self.optimizer.zero_grad()  # 학습 시작 전 그래디언트 초기화
        
        for idx, (inputs, targets) in enumerate(train_loader):
            # 순전파
            outputs = self.model(inputs.cuda())
            loss = self.criterion(outputs, targets.cuda())
            
            # loss를 accumulation_steps로 나누어 계산
            loss = loss / self.accumulation_steps
            loss.backward()
            
            # accumulation_steps만큼 그래디언트 누적 후 업데이트
            if (idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
            running_loss += loss.item()
```

### 2. Mixed Precision Training (혼합 정밀도 학습)

```python
from torch.cuda.amp import autocast, GradScaler

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler()
        
    def train(self, train_loader):
        self.model.train()
        
        for inputs, targets in train_loader:
            inputs = inputs.cuda()
            targets = targets.cuda()
            
            # 자동 혼합 정밀도 적용
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # 스케일링된 역전파
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
```

### 3. 메모리 최적화 기법들

```python
# 1. 메모리 캐시 비우기
def clear_memory():
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

# 2. 모델 메모리 최적화
class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # 레이어 정의
        )
    
    def forward(self, x):
        # 중간 결과물 삭제
        with torch.no_grad():
            x = self.features(x)
        return x

# 3. 데이터로더 최적화
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,  # GPU 전송 속도 향상
    persistent_workers=True  # 워커 재사용
)
```

### 4. 모델 병렬화 (Model Parallelism)

```python
class ModelParallel(nn.Module):
    def __init__(self):
        super(ModelParallel, self).__init__()
        # 첫 번째 GPU에 레이어 할당
        self.layer1 = nn.Linear(1024, 512).cuda(0)
        # 두 번째 GPU에 레이어 할당
        self.layer2 = nn.Linear(512, 256).cuda(1)
        
    def forward(self, x):
        x = x.cuda(0)
        x = self.layer1(x)
        x = x.cuda(1)
        x = self.layer2(x)
        return x
```

### 5. 메모리 효율적인 학습 루프

```python
class EfficientTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scaler = GradScaler()
        
    def train_epoch(self, train_loader, accumulation_steps=4):
        self.model.train()
        running_loss = 0.0
        
        for idx, (inputs, targets) in enumerate(train_loader):
            # 메모리 효율을 위해 배치를 더 작은 크기로 분할
            batch_size = inputs.size(0)
            sub_batch_size = batch_size // accumulation_steps
            
            self.optimizer.zero_grad()
            
            for i in range(0, batch_size, sub_batch_size):
                end = min(i + sub_batch_size, batch_size)
                sub_inputs = inputs[i:end].cuda()
                sub_targets = targets[i:end].cuda()
                
                with autocast():
                    outputs = self.model(sub_inputs)
                    loss = self.criterion(outputs, sub_targets)
                    loss = loss / accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                # 메모리 정리
                del outputs, loss
                torch.cuda.empty_cache()
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item() * accumulation_steps
```

### 6. 추가 메모리 최적화 팁

```python
# 1. 모델 가중치 양자화
from torch.quantization import quantize_dynamic

quantized_model = quantize_dynamic(
    model,
    {nn.Linear, nn.Conv2d},
    dtype=torch.qint8
)

# 2. 체크포인트 사용
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    x = checkpoint(self.feature_extractor, x)
    x = self.classifier(x)
    return x

# 3. 메모리 모니터링
def print_memory_stats():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f}MB")
```

이러한 방법들을 조합하여 사용하면 CUDA 메모리 제한을 극복하고 더 큰 배치 크기로 학습할 수 있습니다. 특히 그래디언트 누적과 혼합 정밀도 학습을 함께 사용하면 매우 효과적입니다.
'''

'''


적은 데이터셋으로도 좋은 성능을 낼 수 있는 세그멘테이션 모델들을 소개해드리겠습니다:

### 1. TransUNet (Transformer + U-Net)

```python
class TransUNet(nn.Module):
    def __init__(self, img_dim=224, in_channels=3, out_channels=1):
        super().__init__()
        
        # Vision Transformer Encoder
        self.vit = ViT(
            img_dim=img_dim,
            patch_dim=16,
            num_channels=in_channels,
            num_heads=12,
            num_layers=12,
            hidden_dim=768,
            dropout_rate=0.1
        )
        
        # U-Net style decoder
        self.decoder = UNetDecoder(
            in_channels=768,
            out_channels=out_channels
        )
        
    def forward(self, x):
        # Transformer encoding
        vit_features = self.vit(x)
        # Decoding with skip connections
        output = self.decoder(vit_features)
        return output
```

**장점:**
- 적은 데이터로도 좋은 성능
- Self-attention으로 전역적 특징 포착
- 사전학습 모델 활용 가능

### 2. Few-Shot U-Net

```python
class FewShotUNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Siamese encoder
        self.encoder = SiameseEncoder()
        
        # Relation module
        self.relation = RelationModule()
        
        # Decoder with attention
        self.decoder = AttentionDecoder()
        
    def forward(self, query_img, support_set):
        # Support set encoding
        support_features = [self.encoder(img) for img in support_set]
        
        # Query image encoding
        query_features = self.encoder(query_img)
        
        # Relation learning
        relation_features = self.relation(query_features, support_features)
        
        # Decoding with attention
        output = self.decoder(relation_features)
        return output
```

**장점:**
- Few-shot learning 가능
- 적은 샘플로 일반화 능력 확보
- 메타러닝 접근 방식

### 3. 데이터 증강 강화 전략

```python
class EnhancedAugmentation:
    def __init__(self):
        self.transform = A.Compose([
            # 기하학적 변환
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.7
            ),
            A.OneOf([
                A.ElasticTransform(
                    alpha=120,
                    sigma=120 * 0.05,
                    alpha_affine=120 * 0.03
                ),
                A.GridDistortion(),
                A.OpticalDistortion(
                    distort_limit=1,
                    shift_limit=0.5
                ),
            ], p=0.3),
            
            # 강도 변환
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.RandomGamma(),
                A.CLAHE(),
            ], p=0.3),
            
            # 노이즈 추가
            A.OneOf([
                A.GaussNoise(),
                A.MultiplicativeNoise(),
                A.ISONoise(),
            ], p=0.2),
            
            # Mixup 및 CutMix
            A.OneOf([
                A.MixUp(p=0.5),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    p=0.5
                ),
            ], p=0.3),
        ])
```

### 4. 준지도 학습 접근법

```python
class SemiSupervisedSegmentation:
    def __init__(self, model, labeled_loader, unlabeled_loader):
        self.model = model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        
    def train_step(self):
        # 레이블된 데이터로 학습
        labeled_loss = self.supervised_forward()
        
        # 레이블되지 않은 데이터로 일관성 정규화
        unlabeled_loss = self.consistency_forward()
        
        # 전체 손실
        total_loss = labeled_loss + 0.5 * unlabeled_loss
        return total_loss
        
    def consistency_forward(self):
        # 약한 증강과 강한 증강 적용
        weak_aug = self.weak_transform(images)
        strong_aug = self.strong_transform(images)
        
        # 일관성 손실 계산
        with torch.no_grad():
            pseudo_labels = self.model(weak_aug)
        pred = self.model(strong_aug)
        
        return consistency_loss(pred, pseudo_labels)
```

### 5. 최적화된 학습 전략

```python
class OptimizedTrainer:
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion
        
        # Cosine Annealing with Warm Restarts
        self.scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2
        )
        
    def train_epoch(self, dataloader):
        for batch in dataloader:
            # Mixed Precision Training
            with autocast():
                outputs = self.model(batch['image'])
                loss = self.criterion(outputs, batch['mask'])
            
            # Gradient Accumulation
            loss = loss / self.accumulation_steps
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
```

### 추천 전략:

1. **데이터가 매우 적은 경우 (100개 미만):**
   - TransUNet + 강력한 데이터 증강
   - Few-Shot U-Net
   - 사전학습 모델 활용

2. **데이터가 어느 정도 있는 경우 (100-500개):**
   - U-Net++ with 강력한 증강
   - 준지도 학습 접근
   - 앙상블 기법 활용

3. **성능 최적화 팁:**
   - 사전학습된 인코더 사용
   - 점진적 학습 (Progressive learning)
   - 교차 검증 활용
   - Test Time Augmentation (TTA)

이러한 접근법들을 조합하여 사용하면 적은 데이터셋으로도 좋은 성능을 얻을 수 있습니다.
'''

'''
 

일반적인 이미지 세그멘테이션 모델의 필요 데이터셋 크기를 분석해드리겠습니다:

### 1. 일반적인 데이터셋 크기 기준

```python
dataset_guidelines = {
    'small': {
        'size': '500-1,000 images',
        'suitable_models': ['U-Net', 'TransUNet'],
        'requirements': 'Strong augmentation needed'
    },
    'medium': {
        'size': '1,000-5,000 images',
        'suitable_models': ['DeepLabV3+', 'HRNet'],
        'requirements': 'Regular augmentation'
    },
    'large': {
        'size': '5,000+ images',
        'suitable_models': ['Mask R-CNN', 'Complex architectures'],
        'requirements': 'Basic augmentation'
    }
}
```

### 2. 모델별 권장 최소 데이터셋 크기

```python
model_requirements = {
    'U-Net': {
        'minimum_samples': 500,
        'optimal_samples': 2000,
        'notes': '데이터 증강으로 300개도 가능'
    },
    'DeepLabV3+': {
        'minimum_samples': 1000,
        'optimal_samples': 3000,
        'notes': '사전학습 모델 필수'
    },
    'Mask R-CNN': {
        'minimum_samples': 2000,
        'optimal_samples': 5000,
        'notes': '복잡한 객체 검출에 적합'
    },
    'TransUNet': {
        'minimum_samples': 400,
        'optimal_samples': 1500,
        'notes': 'Transformer 구조로 적은 데이터로도 가능'
    }
}
```

### 3. 데이터셋 크기별 접근 방법

1. **매우 적은 데이터 (100-500개)**
   ```python
   strategies_for_small_dataset = {
       'augmentation': [
           'Heavy geometric augmentation',
           'Intensity augmentation',
           'Mixup & CutMix',
           'Random erasing'
       ],
       'model_choice': [
           'U-Net with pretrained encoder',
           'TransUNet',
           'Few-shot learning approaches'
       ],
       'training_techniques': [
           'Cross-validation',
           'Transfer learning',
           'Ensemble methods'
       ]
   }
   ```

2. **중간 크기 데이터 (500-2000개)**
   ```python
   strategies_for_medium_dataset = {
       'augmentation': [
           'Moderate augmentation',
           'Random rotation/flip',
           'Color jittering'
       ],
       'model_choice': [
           'U-Net++',
           'DeepLabV3+',
           'HRNet'
       ],
       'training_techniques': [
           'Regular training',
           'Learning rate scheduling',
           'Early stopping'
       ]
   }
   ```

3. **충분한 데이터 (2000개 이상)**
   ```python
   strategies_for_large_dataset = {
       'augmentation': [
           'Basic augmentation',
           'Optional heavy augmentation'
       ],
       'model_choice': [
           'Complex architectures',
           'Custom models',
           'Ensemble of multiple models'
       ],
       'training_techniques': [
           'Regular training',
           'Multi-scale training',
           'Advanced optimization'
       ]
   }
   ```

### 4. 데이터셋 크기 최적화 팁

```python
optimization_tips = {
    'data_quality': [
        '품질 좋은 annotation',
        '다양한 케이스 포함',
        '균형잡힌 클래스 분포'
    ],
    'preprocessing': [
        '적절한 해상도 선택',
        '노이즈 제거',
        '정규화'
    ],
    'augmentation': [
        '도메인 특성에 맞는 증강',
        '현실적인 변형',
        '클래스 균형 고려'
    ]
}
```

### 결론:

1. **최소 권장 데이터셋 크기:**
   - 기본: 500-1000개
   - 복잡한 문제: 2000개 이상
   - 단순한 문제: 300-500개도 가능

2. **성공적인 학습을 위한 조건:**
   - 데이터의 품질
   - 적절한 증강 기법
   - 문제에 맞는 모델 선택
   - 효과적인 학습 전략

3. **데이터셋 크기 보완 방법:**
   - 강력한 데이터 증강
   - 전이 학습
   - 준지도 학습
   - 앙상블 기법

실제 필요한 데이터셋 크기는 문제의 복잡도, 요구되는 정확도, 사용할 수 있는 증강 기법 등에 따라 달라질 수 있습니다.
'''