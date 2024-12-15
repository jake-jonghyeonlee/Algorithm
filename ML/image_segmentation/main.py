import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
from pathlib import Path
from models import UNet, AttentionUNet, ResUNet, DenseUNet, SEUNet, DeepLabV3Plus
from augmentation import get_transforms
from loss_function import get_loss_function
from dataset import PetSegmentationDataset

def get_dataloaders(batch_size, train_transform, valid_transform):
    # 데이터셋 경로 설정
    data_root = Path("data/oxford-iiit-pet")
    data_root.mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 생성
    train_dataset = PetSegmentationDataset(
        root=str(data_root),
        transform=train_transform,
        train=True
    )
    
    valid_dataset = PetSegmentationDataset(
        root=str(data_root),
        transform=valid_transform,
        train=False
    )
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, valid_loader, train_dataset, valid_dataset

def objective(trial):
    # 모델 선택
    model_type = trial.suggest_categorical("model_type", 
                                         ["UNet", "AttentionUNet", "ResUNet", 
                                          "DenseUNet", "SEUNet", "DeepLabV3Plus"])
    
    # 학습 하이퍼파라미터
    lr_params = {
        # 초기 learning rate
        'lr': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        # learning rate scheduler 관련 파라미터
        'scheduler_patience': trial.suggest_int("scheduler_patience", 2, 5),
        'scheduler_factor': trial.suggest_float("scheduler_factor", 0.1, 0.5),
        # optimizer 관련 파라미터
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        'beta1': trial.suggest_float("beta1", 0.8, 0.99),
        'beta2': trial.suggest_float("beta2", 0.9, 0.999),
    }
    
    batch_size = trial.suggest_int("batch_size", 8, 32)
    
    # Augmentation 하이퍼파라미터 설정
    aug_params = {
        # 각 augmentation의 사용 여부
        'horizontal_flip': trial.suggest_categorical('horizontal_flip', [True, False]),
        'vertical_flip': trial.suggest_categorical('vertical_flip', [True, False]),
        'rotate': trial.suggest_categorical('rotate', [True, False]),
        'brightness_contrast': trial.suggest_categorical('brightness_contrast', [True, False]),
        
        # 각 augmentation의 확률
        'horizontal_flip_p': trial.suggest_float('horizontal_flip_p', 0.0, 1.0) if trial.suggest_categorical('horizontal_flip', [True, False]) else 0.0,
        'vertical_flip_p': trial.suggest_float('vertical_flip_p', 0.0, 1.0) if trial.suggest_categorical('vertical_flip', [True, False]) else 0.0,
        'rotate_p': trial.suggest_float('rotate_p', 0.0, 1.0) if trial.suggest_categorical('rotate', [True, False]) else 0.0,
        'brightness_contrast_p': trial.suggest_float('brightness_contrast_p', 0.0, 1.0) if trial.suggest_categorical('brightness_contrast', [True, False]) else 0.0,
        
        # brightness_contrast의 추가 파라미터
        'brightness_limit': trial.suggest_float('brightness_limit', 0.1, 0.3) if trial.suggest_categorical('brightness_contrast', [True, False]) else 0.2,
        'contrast_limit': trial.suggest_float('contrast_limit', 0.1, 0.3) if trial.suggest_categorical('brightness_contrast', [True, False]) else 0.2,
    }
    
    # Loss function 하이퍼파라미터 설정
    loss_params = {
        'loss_type': trial.suggest_categorical('loss_type', 
                                             ['bce', 'dice', 'focal', 
                                              'tversky', 'iou', 'combo']),
    }
    
    # loss type에 따른 추가 파라미터
    if loss_params['loss_type'] == 'dice' or loss_params['loss_type'] == 'iou':
        loss_params['smooth'] = trial.suggest_float('smooth', 0.1, 2.0)
    elif loss_params['loss_type'] == 'focal':
        loss_params['alpha'] = trial.suggest_float('focal_alpha', 0.1, 0.9)
        loss_params['gamma'] = trial.suggest_float('focal_gamma', 1.0, 4.0)
    elif loss_params['loss_type'] == 'tversky':
        loss_params['alpha'] = trial.suggest_float('tversky_alpha', 0.2, 0.8)
        loss_params['beta'] = trial.suggest_float('tversky_beta', 0.2, 0.8)
        loss_params['smooth'] = trial.suggest_float('smooth', 0.1, 2.0)
    elif loss_params['loss_type'] == 'combo':
        loss_params['bce_weight'] = trial.suggest_float('bce_weight', 0.0, 1.0)
        loss_params['dice_weight'] = trial.suggest_float('dice_weight', 0.0, 1.0)
        loss_params['focal_weight'] = trial.suggest_float('focal_weight', 0.0, 1.0)
    
    # Transform 및 데이터 로더 설정
    train_transform = get_transforms(train=True, aug_params=aug_params)
    valid_transform = get_transforms(train=False)
    train_loader, valid_loader, _, _ = get_dataloaders(batch_size, train_transform, valid_transform)
    
    # 모델 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "UNet":
        model = UNet().to(device)
    elif model_type == "AttentionUNet":
        model = AttentionUNet().to(device)
    elif model_type == "ResUNet":
        model = ResUNet().to(device)
    elif model_type == "DenseUNet":
        model = DenseUNet().to(device)
    elif model_type == "DeepLabV3Plus":
        model = DeepLabV3Plus().to(device)
    else:
        model = SEUNet().to(device)
    
    # Loss function 설정
    criterion = get_loss_function(loss_params)
    
    # 최적화 도구 설정
    optimizer = optim.Adam(
        model.parameters(), 
        lr=lr_params['lr'],
        weight_decay=lr_params['weight_decay'],
        betas=(lr_params['beta1'], lr_params['beta2'])
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min',
        patience=lr_params['scheduler_patience'],
        factor=lr_params['scheduler_factor']
    )
    
    # 학습
    best_valid_loss = float('inf')
    for epoch in range(10):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # 검증
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            
    return best_valid_loss

def main():
    # Optuna 최적화
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    # 최적의 파라미터로 transform 설���
    aug_params = {k: v for k, v in best_params.items() 
                 if k in ['horizontal_flip', 'vertical_flip', 'rotate', 
                         'brightness_contrast', 'horizontal_flip_p', 
                         'vertical_flip_p', 'rotate_p', 'brightness_contrast_p',
                         'brightness_limit', 'contrast_limit']}
    
    train_transform = get_transforms(train=True, aug_params=aug_params)
    valid_transform = get_transforms(train=False)
    
    # 데이터 로더 설정
    train_loader, valid_loader, train_dataset, valid_dataset = get_dataloaders(
        batch_size=best_params["batch_size"],
        train_transform=train_transform,
        valid_transform=valid_transform
    )
    
    # Loss function 설정
    loss_params = {k: v for k, v in best_params.items() 
                  if k in ['loss_type', 'smooth', 'focal_alpha', 'focal_gamma',
                          'tversky_alpha', 'tversky_beta', 'bce_weight',
                          'dice_weight', 'focal_weight']}
    criterion = get_loss_function(loss_params)
    
    # 최적의 모델 선정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if best_params["model_type"] == "UNet":
        model = UNet().to(device)
    elif best_params["model_type"] == "AttentionUNet":
        model = AttentionUNet().to(device)
    elif best_params["model_type"] == "ResUNet":
        model = ResUNet().to(device)
    elif best_params["model_type"] == "DenseUNet":
        model = DenseUNet().to(device)
    elif best_params["model_type"] == "DeepLabV3Plus":
        model = DeepLabV3Plus().to(device)
    else:
        model = SEUNet().to(device)
    
    # 최적화 도구 설정
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        betas=(best_params["beta1"], best_params["beta2"])
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=best_params["scheduler_patience"],
        factor=best_params["scheduler_factor"]
    )
    
    # 모델 저장 경로
    save_path = Path("models")
    save_path.mkdir(exist_ok=True)
    
    # 학습 실행
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), 
                      save_path / f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()

'''
최적화 시간을 단축하기 위한 몇 가지 전략을 제안드립니다:

1. **탐색 공간 축소**:
```python
def objective(trial):
    # 모델 타입 축소
    model_type = trial.suggest_categorical("model_type", 
                                         ["UNet", "ResUNet"])  # 가장 기본적인 모델만
    
    # 학습 파라미터 범위 축소
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)  # 일반적으로 효과적인 범위로 축소
    batch_size = trial.suggest_int("batch_size", 16, 32)   # 더 좁은 범위
    
    # 핵심 augmentation만 선택
    aug_params = {
        'horizontal_flip': True,  # 고정
        'vertical_flip': False,   # 고정
        'rotate': trial.suggest_categorical('rotate', [True, False]),
        'brightness_contrast': trial.suggest_categorical('brightness_contrast', [True, False]),
    }
    
    # loss function 단순화
    loss_params = {
        'loss_type': trial.suggest_categorical('loss_type', ['bce', 'dice'])  # 주요 loss만
    }
```

2. **학습 에폭 수 감소**:
```python
def objective(trial):
    # 에폭 수를 줄여서 각 trial의 실행 시간 단축
    for epoch in range(3):  # 10 -> 3
        model.train()
        train_loss = 0
        # ... 학습 코드 ...
```

3. **데이터셋 크기 축소**:
```python
class PetSegmentationDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.dataset = OxfordIIITPet(root=root, 
                                   download=True,
                                   target_types="segmentation")
        self.transform = transform
        self.train = train
        
        # 전체 데이터의 일부만 사용
        n_samples = len(self.dataset)
        if train:
            self.indices = range(int(0.2 * n_samples))  # 20%만 사용
        else:
            self.indices = range(int(0.8 * n_samples), int(0.85 * n_samples))  # 검증도 일부만
```

4. **병렬 처리 최적화**:
```python
def main():
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=5),  # 초기 랜덤 시도 수 감소
        study_name="quick_optimization"
    )
    
    # 병렬 처리를 위한 설정
    study.optimize(objective, 
                  n_trials=10,  # trial 수 감소
                  n_jobs=-1)    # 가능한 모든 CPU 코어 사용
```

5. **조기 종료 설정**:
```python
def objective(trial):
    # 성능이 좋지 않은 trial 조기 종료
    pruner = optuna.pruners.MedianPruner()
    
    for epoch in range(3):
        # ... 학습 코드 ...
        
        # 현재 trial의 성능이 다른 trial들의 중간값보다 많이 나쁘면 종료
        trial.report(valid_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
```

6. **하드웨어 최적화**:
```python
def get_dataloaders(batch_size, train_transform, valid_transform):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,  # CPU 코어 수에 따라 조정
        pin_memory=True
    )
```

이러한 최적화를 적용하면 전체 최적화 시간을 크게 줄일 수 있습니다. 초기 빠른 최적화 후, 찾은 최적 파라미터 근처에서 더 세밀한 탐색을 수행하는 2단계 접근도 고려해볼 수 있습니다.
'''