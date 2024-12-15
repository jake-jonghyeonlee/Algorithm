import torch
import numpy as np
import cv2
from PIL import Image
import random

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image, mask=None):
        for transform in self.transforms:
            image, mask = transform(image, mask)
        return image, mask

class ToTensor:
    def __call__(self, image, mask=None):
        # PIL Image나 numpy array를 tensor로 변환
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # HWC -> CHW
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        
        if mask is not None:
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            mask = torch.from_numpy(mask).float()
            
        return image, mask

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
        
    def __call__(self, image, mask=None):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image, mask

class RandomResizedCrop:
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.scale = scale
        
    def __call__(self, image, mask=None):
        height, width = image.shape[:2]
        
        # 랜덤 크기와 비율 선택
        area = height * width
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(3/4, 4/3)
        
        w = int(round(np.sqrt(target_area * aspect_ratio)))
        h = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if w <= width and h <= height:
            x = random.randint(0, width - w)
            y = random.randint(0, height - h)
            
            image = image[y:y+h, x:x+w]
            if mask is not None:
                mask = mask[y:y+h, x:x+w]
        
        # 지정된 크기로 리사이즈
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            
        return image, mask

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask=None):
        if random.random() < self.p:
            image = np.fliplr(image)
            if mask is not None:
                mask = np.fliplr(mask)
        return image, mask

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, image, mask=None):
        if random.random() < self.p:
            image = np.flipud(image)
            if mask is not None:
                mask = np.flipud(mask)
        return image, mask

class RandomRotate:
    def __init__(self, p=0.5, angles=[0, 90, 180, 270]):
        self.p = p
        self.angles = angles
        
    def __call__(self, image, mask=None):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            if angle == 0:
                return image, mask
                
            height, width = image.shape[:2]
            matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
            image = cv2.warpAffine(image, matrix, (width, height))
            
            if mask is not None:
                mask = cv2.warpAffine(mask, matrix, (width, height))
                
        return image, mask

class RandomBrightnessContrast:
    def __init__(self, p=0.5, brightness_limit=0.2, contrast_limit=0.2):
        self.p = p
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        
    def __call__(self, image, mask=None):
        if random.random() < self.p:
            alpha = 1.0 + random.uniform(-self.contrast_limit, self.contrast_limit)
            beta = random.uniform(-self.brightness_limit, self.brightness_limit)
            
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta * 255)
            
        return image, mask

class Resize:
    def __init__(self, size):
        self.size = size if isinstance(size, tuple) else (size, size)
        
    def __call__(self, image, mask=None):
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        return image, mask

def get_transforms(train=True, aug_params=None):
    if train:
        transforms = []
        
        # 기본 RandomResizedCrop은 항상 포함
        transforms.append(RandomResizedCrop(256, scale=(0.8, 1.0)))
        
        # 각 augmentation의 적용 여부를 aug_params로 제어
        if aug_params:
            if aug_params.get('horizontal_flip', False):
                transforms.append(RandomHorizontalFlip(p=aug_params.get('horizontal_flip_p', 0.5)))
                
            if aug_params.get('vertical_flip', False):
                transforms.append(RandomVerticalFlip(p=aug_params.get('vertical_flip_p', 0.5)))
                
            if aug_params.get('rotate', False):
                transforms.append(RandomRotate(p=aug_params.get('rotate_p', 0.5)))
                
            if aug_params.get('brightness_contrast', False):
                transforms.append(RandomBrightnessContrast(
                    p=aug_params.get('brightness_contrast_p', 0.5),
                    brightness_limit=aug_params.get('brightness_limit', 0.2),
                    contrast_limit=aug_params.get('contrast_limit', 0.2)
                ))
        
        # 기본 Normalize와 ToTensor는 항상 포함
        transforms.extend([Normalize(), ToTensor()])
        return Compose(transforms)
    else:
        return Compose([
            Resize(256),
            Normalize(),
            ToTensor(),
        ]) 