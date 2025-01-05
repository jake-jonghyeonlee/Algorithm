import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageDraw
import numpy as np

class Augmentation:
    def __init__(self, output_size=512):
        self.output_size = output_size

    def resize(self, image, mask):
        image = TF.resize(image, (self.output_size, self.output_size))
        mask = TF.resize(mask, (self.output_size, self.output_size))
        return image, mask

    def random_crop(self, image, mask):
        i, j, h, w = TF.get_random_crop_params(image, (self.output_size, self.output_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return image, mask

    def horizontal_flip(self, image, mask, p=0.5):
        if random.random() < p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask

    def vertical_flip(self, image, mask, p=0.5):
        if random.random() < p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask

    def rotate(self, image, mask, p=0.5):
        if random.random() < p:
            angle = random.choice([0, 90, 180, 270])
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        return image, mask

    def adjust_brightness(self, image, mask, p=0.5):
        if random.random() < p:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        return image, mask

    def add_random_lines(self, image, mask, p=0.3):
        """Add random lines to create pattern breaks using PyTorch tensors"""
        if random.random() < p:
            # Convert to tensor if not already
            if not isinstance(image, torch.Tensor):
                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

            # Create line mask
            line_mask = torch.zeros((1, self.output_size, self.output_size), device=image.device)
            
            num_lines = random.randint(1, 3)
            for _ in range(num_lines):
                start_x = random.randint(0, self.output_size-1)
                start_y = random.randint(0, self.output_size-1)
                end_x = random.randint(0, self.output_size-1)
                end_y = random.randint(0, self.output_size-1)
                
                # Create line using linear interpolation
                steps = max(abs(end_x - start_x), abs(end_y - start_y)) * 2
                if steps > 0:
                    x_coords = torch.linspace(start_x, end_x, steps)
                    y_coords = torch.linspace(start_y, end_y, steps)
                    
                    # Draw thick line
                    thickness = random.randint(2, 5)
                    for dx in range(-thickness//2, thickness//2 + 1):
                        for dy in range(-thickness//2, thickness//2 + 1):
                            x = (x_coords + dx).long().clamp(0, self.output_size-1)
                            y = (y_coords + dy).long().clamp(0, self.output_size-1)
                            line_mask[0, y, x] = 1

            # Apply random intensity to lines
            image = image * (1 - line_mask) + torch.rand_like(image) * line_mask
            mask = torch.max(mask, line_mask)

            if not isinstance(image, torch.Tensor):
                image = TF.to_pil_image(image)
                mask = Image.fromarray(mask.squeeze().numpy().astype(np.uint8))

        return image, mask

    def add_noise_patches(self, image, mask, p=0.3):
        """Add noise patches using PyTorch operations"""
        if random.random() < p:
            if not isinstance(image, torch.Tensor):
                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

            num_patches = random.randint(1, 3)
            for _ in range(num_patches):
                x = random.randint(0, self.output_size - 20)
                y = random.randint(0, self.output_size - 20)
                size = random.randint(10, 20)
                
                # Create noise patch
                noise = torch.rand(3, size, size, device=image.device)
                patch_mask = torch.ones(1, size, size, device=image.device)
                
                # Apply patch
                image[:, y:y+size, x:x+size] = noise
                mask[:, y:y+size, x:x+size] = patch_mask

            if not isinstance(image, torch.Tensor):
                image = TF.to_pil_image(image)
                mask = Image.fromarray(mask.squeeze().numpy().astype(np.uint8))

        return image, mask

    def add_pattern_breaks(self, image, mask, p=0.3):
        """Add pattern breaks using PyTorch operations"""
        if random.random() < p:
            if not isinstance(image, torch.Tensor):
                image = TF.to_tensor(image)
                mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

            break_type = random.choice(['gap', 'blur', 'intensity'])
            
            x = random.randint(0, self.output_size - 30)
            y = random.randint(0, self.output_size - 30)
            size = random.randint(15, 30)
            
            if break_type == 'gap':
                image[:, y:y+size, x:x+size] = 0
                mask[:, y:y+size, x:x+size] = 1
                
            elif break_type == 'blur':
                # Apply Gaussian blur to region
                region = image[:, y:y+size, x:x+size].unsqueeze(0)
                blurred = F.gaussian_blur(region, kernel_size=5, sigma=2.0)
                image[:, y:y+size, x:x+size] = blurred.squeeze(0)
                mask[:, y:y+size, x:x+size] = 1
                
            else:  # intensity
                intensity_factor = random.uniform(0.5, 1.5)
                image[:, y:y+size, x:x+size] *= intensity_factor
                mask[:, y:y+size, x:x+size] = 1

            if not isinstance(image, torch.Tensor):
                image = TF.to_pil_image(image)
                mask = Image.fromarray(mask.squeeze().numpy().astype(np.uint8))

        return image, mask

    def normalize(self, image):
        return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def to_tensor(self, image, mask):
        image = TF.to_tensor(image)
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask

    def __call__(self, image, mask, train=True):
        image, mask = self.resize(image, mask)
        
        if train:
            image, mask = self.horizontal_flip(image, mask)
            image, mask = self.vertical_flip(image, mask)
            image, mask = self.rotate(image, mask)
            image, mask = self.adjust_brightness(image, mask)
            
            # Apply pattern break augmentations
            image, mask = self.add_random_lines(image, mask)
            image, mask = self.add_noise_patches(image, mask)
            image, mask = self.add_pattern_breaks(image, mask)
        
        image, mask = self.to_tensor(image, mask)
        image = self.normalize(image)
        
        return image, mask
