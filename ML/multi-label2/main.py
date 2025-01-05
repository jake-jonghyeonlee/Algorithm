import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from augmentation import Augmentation
from model import PatternAwareUNet
from dataset import SegmentationDataset

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 클래스별 손실 계산
            loss = 0
            for cls in range(masks.shape[1]):
                loss += criterion(outputs[:, cls:cls+1, :, :], masks[:, cls:cls+1, :, :])
            loss = loss / masks.shape[1]
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

def main():
    # 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 4  # 클래스 수 설정
    batch_size = 8
    learning_rate = 1e-4
    num_epochs = 100
    
    # 모델 초기화
    model = PatternAwareUNet(in_channels=3, num_classes=num_classes).to(device)
    
    # Loss 함수 정의
    from loss import get_loss_function
    criterion = get_loss_function('combined', dice_weight=0.5, focal_weight=0.5)
    
    # Optimizer와 Scheduler 설정
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 데이터셋 경로 설정 (실제 경로로 수정 필요)
    train_image_paths = []  # 학습 이미지 경로 리스트
    train_mask_paths = []   # 학습 마스크 경로 리스트
    
    # 데이터셋과 데이터로더 설정
    train_dataset = SegmentationDataset(
        train_image_paths, 
        train_mask_paths,
        num_classes=num_classes,
        train=True,
        patch_size=256
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # 모델 학습
    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs)

if __name__ == "__main__":
    main()
