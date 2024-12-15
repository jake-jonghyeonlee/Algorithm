import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path

class CustomSegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None, train=True):
        """
        Args:
            data_dir (str): 데이터 디렉토리 경로
            transform: 데이터 증강을 위한 transform
            train (bool): 학습/검증 모드 설정
        
        디렉토리 구조:
        data_dir/
            ├── images/         # 원본 이미지
            │   ├── train/      # 학습용 이미지
            │   └── valid/      # 검증용 이미지
            └── masks/          # 마스크 이미지
                ├── train/      # 학습용 마스크
                └── valid/      # 검증용 마스크
        """
        self.transform = transform
        self.train = train
        
        # 경로 설정
        base_dir = Path(data_dir)
        img_dir = base_dir / 'images' / ('train' if train else 'valid')
        mask_dir = base_dir / 'masks' / ('train' if train else 'valid')
        
        # 이미지 파일 리스트 생성
        self.image_paths = sorted(list(img_dir.glob('*.png')))  # 또는 *.jpg
        self.mask_paths = sorted(list(mask_dir.glob('*.png')))
        
        assert len(self.image_paths) == len(self.mask_paths), \
            "이미지와 마스크의 개수가 일치하지 않습니다!"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Tips for better segmentation:
        1. 이미지 읽기 시 cv2.IMREAD_COLOR 사용하여 BGR로 읽은 후 RGB로 변환
        2. 마스크는 그레이스케일로 읽어서 이진화
        3. 이미지 정규화 시 mean과 std 값을 데이터셋에 맞게 계산하여 사용
        4. 마스크값이 0과 1만 있는지 확인
        """
        # 이미지 로드
        img_path = str(self.image_paths[idx])
        mask_path = str(self.mask_paths[idx])
        
        # BGR로 읽은 후 RGB로 변환
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 마스크는 그레이스케일로 읽기
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 마스크 이진화 (임계값 127 사용, 필요시 조정)
        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        
        # float32로 변환
        image = image.astype(np.float32)
        mask = mask.astype(np.float32)
        
        # Transform 적용
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0)  # mask를 [1, H, W] 형태로 변환

    @staticmethod
    def calculate_mean_std(data_dir):
        """
        데이터셋의 mean과 std 계산
        정확한 정규화를 위해 사용
        """
        img_dir = Path(data_dir) / 'images' / 'train'
        images = []
        
        for img_path in img_dir.glob('*.png'):  # 또는 *.jpg
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
        
        images = np.stack(images, axis=0)
        mean = np.mean(images, axis=(0, 1, 2)) / 255.0
        std = np.std(images, axis=(0, 1, 2)) / 255.0
        
        return mean, std