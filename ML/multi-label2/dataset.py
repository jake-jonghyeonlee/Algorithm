import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from augmentation import Augmentation

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, num_classes=4, train=True, patch_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.num_classes = num_classes
        self.train = train
        self.patch_size = patch_size
        self.augmentation = Augmentation(output_size=512)

    def extract_patches(self, image, mask):
        """이미지와 마스크를 패치로 분할"""
        patches_img = []
        patches_mask = []
        h, w = image.size
        
        for i in range(0, h-self.patch_size+1, self.patch_size//2):
            for j in range(0, w-self.patch_size+1, self.patch_size//2):
                patch_img = image.crop((i, j, i+self.patch_size, j+self.patch_size))
                patch_mask = mask.crop((i, j, i+self.patch_size, j+self.patch_size))
                
                # 패치에 패턴이 충분히 포함된 경우만 사용
                mask_np = np.array(patch_mask)
                if np.any(mask_np > 0):  # 패턴이 있는 경우
                    patches_img.append(patch_img)
                    patches_mask.append(patch_mask)
        
        return patches_img, patches_mask

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        mask = Image.open(self.mask_paths[idx])
        
        if self.train:
            # 패치 추출
            patches_img, patches_mask = self.extract_patches(image, mask)
            
            # 랜덤하게 패치 선택
            if patches_img:
                patch_idx = np.random.randint(len(patches_img))
                image = patches_img[patch_idx]
                mask = patches_mask[patch_idx]
        
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

    def __len__(self):
        return len(self.image_paths)
