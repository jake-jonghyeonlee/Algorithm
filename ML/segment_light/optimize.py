import torch
import torch.nn as nn
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
import numpy as np
from loss_function import get_loss_function
from evaluation import SegmentationMetrics, print_metrics

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class LightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 더모리 효율을 위해 초기 채널 수를 16으로 시작
        self.init_channels = 16
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, self.init_channels)  # 16
        self.enc2 = DoubleConv(self.init_channels, self.init_channels*2)  # 32
        self.enc3 = DoubleConv(self.init_channels*2, self.init_channels*4)  # 64
        self.enc4 = DoubleConv(self.init_channels*4, self.init_channels*8)  # 128
        self.enc5 = DoubleConv(self.init_channels*8, self.init_channels*16)  # 256
        
        # Bridge
        self.bridge = DoubleConv(self.init_channels*16, self.init_channels*32)  # 512
        
        # Decoder - 메모리 효율을 위해 점진적으로 채널 수 감소
        self.up1 = nn.ConvTranspose2d(self.init_channels*32, self.init_channels*16, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(self.init_channels*32, self.init_channels*16)  # 256
        
        self.up2 = nn.ConvTranspose2d(self.init_channels*16, self.init_channels*8, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.init_channels*16, self.init_channels*8)  # 128
        
        self.up3 = nn.ConvTranspose2d(self.init_channels*8, self.init_channels*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.init_channels*8, self.init_channels*4)  # 64
        
        self.up4 = nn.ConvTranspose2d(self.init_channels*4, self.init_channels*2, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.init_channels*4, self.init_channels*2)  # 32
        
        # Final layer
        self.final_conv = nn.Conv2d(self.init_channels*2, out_channels, kernel_size=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder path with skip connections
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))
        
        # Bridge
        bridge = self.bridge(self.pool(enc5))
        
        # Decoder path
        # 메모리 효율을 위해 순차적으로 처리하고 불필요한 텐서는 즉시 해제
        dec1 = self.dec1(torch.cat([self.up1(bridge), enc5], dim=1))
        del bridge, enc5  # 불필요한 텐서 해제
        
        dec2 = self.dec2(torch.cat([self.up2(dec1), enc4], dim=1))
        del dec1, enc4
        
        dec3 = self.dec3(torch.cat([self.up3(dec2), enc3], dim=1))
        del dec2, enc3
        
        dec4 = self.dec4(torch.cat([self.up4(dec3), enc2], dim=1))
        del dec3, enc2
        
        # Final convolution
        output = self.final_conv(dec4)
        return torch.sigmoid(output)

def train_model(data_dir, num_epochs=50, batch_size=2, learning_rate=0.001, 
                num_workers=2, optimizer_params=None, loss_params=None):
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 생성
    dataset = SegmentationDataset(data_dir)
    
    # train/validation 분할
    train_idx, val_idx = train_test_split(
        list(range(len(dataset))), 
        test_size=0.2, 
        random_state=42
    )
    
    # DataLoader 생성
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 모델 초기화
    model = LightUNet().to(device)
    
    # optimizer 파라미터 설정
    if optimizer_params is None:
        optimizer_params = {'lr': learning_rate}
    
    # 옵티마이저 정의
    optimizer = optim.Adam(model.parameters(), **optimizer_params)
    
    # 메모리 효율을 위한 mixed precision 학습 설정
    scaler = torch.cuda.amp.GradScaler()
    
    # 손실 함수 정의
    criterion = get_loss_function('combo', **loss_params) if loss_params else get_loss_function('combo')
    
    # 옵고 성능 초기화 (validation dice score 기준)
    best_dice_score = 0.0
    
    # 평가 메트릭 초기화
    metrics = SegmentationMetrics()
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # mixed precision 학습
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # 메트릭 계산을 위해 예측값과 실제값 저장
            train_predictions.append(outputs.cpu())
            train_targets.append(masks.cpu())
            
            # 메모리 캐시 정리
            torch.cuda.empty_cache()
        
        # 검증 모드
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                
                # 메트릭 계산을 위해 예측값과 실제값 저장
                val_predictions.append(outputs.cpu())
                val_targets.append(masks.cpu())
        
        # 손실값 평균 계산
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # 전체 예측값과 실제값 결합
        train_predictions = torch.cat(train_predictions)
        train_targets = torch.cat(train_targets)
        val_predictions = torch.cat(val_predictions)
        val_targets = torch.cat(val_targets)
        
        # 메트릭 계산
        train_metrics = metrics.get_metrics(train_predictions, train_targets)
        val_metrics = metrics.get_metrics(val_predictions, val_targets)
        
        # 결과 출력
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print_metrics(train_metrics, prefix="Training")
        print_metrics(val_metrics, prefix="Validation")
        
        # 모델 저장 (검증 Dice Score 기준)
        current_dice_score = val_metrics['Dice Coefficient']
        if current_dice_score > best_dice_score:
            best_dice_score = current_dice_score
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'best_dice_score': best_dice_score
            }, 'best_model.pkl')
            print(f'Model saved! Best Dice Score: {best_dice_score:.4f}')
        
        print('=' * 80)
    
    return model

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): 데이터가 있는 디렉토리 경로
                           구조: data_dir/
                                    ├── images/
                                    │   └── *.jpg
                                    └── masks/
                                        └── *.png
            transform: 이미지 변환을 위한 transform 함수
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 이미지와 마스크 경로
        self.image_dir = os.path.join(data_dir, 'images')
        self.mask_dir = os.path.join(data_dir, 'masks')
        
        # jpg 이미지 파일만 리스트에 포함
        self.images = sorted([f for f in os.listdir(self.image_dir) 
                            if f.endswith('.jpg')])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 이미지 경로 (.jpg)
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 마스크 경로 (.png) - 파일명만 가져와 확장자만 변경
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 이미지와 마스크 로드
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # grayscale로 변환
        
        # 기본 transform 적용 (1024x1024로 리사이즈)
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        image = transform(image)
        mask = transform(mask)
        
        # 추가 transform이 있다면 적용
        if self.transform:
            image = self.transform(image)
        
        return image, mask

def get_dataloader(data_dir, batch_size=4, shuffle=True, num_workers=4):
    """
    데이터로더를 생성하는 헬퍼 함수
    Args:
        data_dir: 데이터셋 디렉토리 경로
        batch_size: 배치 크기
        shuffle: 데이터 셔플 여부
        num_workers: 데이터 로딩에 사용할 워커 수
    Returns:
        DataLoader 인스턴스
    """
    dataset = SegmentationDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    return dataloader
