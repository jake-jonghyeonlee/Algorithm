import torch
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path

class PetSegmentationDataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        self.dataset = OxfordIIITPet(root=root, 
                                   download=True,
                                   target_types="segmentation")
        self.transform = transform
        self.train = train
        
        # 학습/검증 분할 (80/20)
        n_samples = len(self.dataset)
        if train:
            self.indices = range(int(0.8 * n_samples))
        else:
            self.indices = range(int(0.8 * n_samples), n_samples)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        idx = self.indices[idx]
        image, mask = self.dataset[idx]
        
        # PIL Image를 NumPy 배열로 변환
        image = np.array(image)
        mask = np.array(mask)
        mask = (mask == 2).astype(np.float32)  # 전경(동물)만 1로 설정
            
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            # 만약 augmented가 튜플을 반환한다면:
            if isinstance(augmented, tuple):
                image, mask = augmented  # 튜플 언패킹
            else:
                # 딕셔너리를 반환한다면:
                image = augmented['image']
                mask = augmented['mask']
        
        return image, mask.unsqueeze(0)  # mask를 [1, H, W] 형태로 변환 