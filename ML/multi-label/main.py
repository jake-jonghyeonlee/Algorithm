import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from augmentation import Augmentation

# Augmentation pipeline
def get_transforms(train=True):
    if train:
        transform = Augmentation(output_size=512)
    else:
        transform = Augmentation(output_size=512, train=False)
    return transform

# Double convolution block for U-Net
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

# U-Net model
class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4):  
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.enc5 = DoubleConv(512, 1024)
        
        # Decoder
        self.dec4 = DoubleConv(1024 + 512, 512)
        self.dec3 = DoubleConv(512 + 256, 256)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.dec1 = DoubleConv(128 + 64, 64)
        
        # 각 클래스별 출력을 위한 최종 컨볼루션
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        x = self.pool(enc1)
        
        enc2 = self.enc2(x)
        x = self.pool(enc2)
        
        enc3 = self.enc3(x)
        x = self.pool(enc3)
        
        enc4 = self.enc4(x)
        x = self.pool(enc4)
        
        x = self.enc5(x)
        
        # Decoder
        x = self.up(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.dec4(x)
        
        x = self.up(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.dec3(x)
        
        x = self.up(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.dec2(x)
        
        x = self.up(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.dec1(x)
        
        return self.final_conv(x)

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes=4, train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.train = train
        self.augmentation = Augmentation(output_size=512)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        
        # 마스크를 원-핫 인코딩으로 변환
        mask = np.array(mask)
        masks = [(mask == i).astype(np.float32) for i in range(self.num_classes)]
        mask = np.stack(masks, axis=0)
        mask = torch.from_numpy(mask)
        
        if self.train:
            image, mask = self.augmentation(image, mask, train=True)
        else:
            image, mask = self.augmentation(image, mask, train=False)
            
        return image, mask

# Training configuration
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=100):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # 각 클래스별로 손실 계산
            loss = 0
            for cls in range(masks.shape[1]):
                loss += criterion(outputs[:, cls:cls+1, :, :], masks[:, cls:cls+1, :, :])
            loss = loss / masks.shape[1]  # 평균 손실
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}')

# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_classes = 4  
    
    # Initialize model
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    from loss import get_loss_function
    
    # Dice Loss가 멀티클래스에 효과적입니다
    criterion = get_loss_function('dice', smooth=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 데이터 로더 설정
    # train_dataset = SegmentationDataset(image_paths, mask_paths, num_classes=num_classes, train=True)
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Train the model
    # train_model(model, train_loader, criterion, optimizer, device)