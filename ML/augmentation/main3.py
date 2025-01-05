def pattern_break_augmentation(image, mask=None):
    """패턴이 끊어진 효과를 주는 augmentation"""
    augmented_images = []
    augmented_masks = [] if mask is not None else None
    
    # 원본 추가
    augmented_images.append(image.copy())
    if mask is not None:
        augmented_masks.append(mask.copy())
    
    height, width = image.shape[:2]
    
    # 1. Random Cutout (작은 직사각형 영역을 제거)
    def apply_cutout(img, max_cuts=3):
        img = img.copy()
        for _ in range(np.random.randint(1, max_cuts + 1)):
            # 작은 크기의 직사각형만 생성
            h = np.random.randint(2, 5)
            w = np.random.randint(2, 5)
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            
            # 해당 영역을 주변 픽셀의 평균값으로 채움
            surrounding = img[max(0, y-1):min(height, y+h+1),
                           max(0, x-1):min(width, x+w+1)]
            fill_value = np.mean(surrounding)
            img[y:y+h, x:x+w] = fill_value
        return img
    
    # 2. Random Noise Injection (특정 영역에만 노이즈 추가)
    def apply_noise(img, noise_prob=0.1):
        img = img.copy()
        noise_mask = np.random.random(img.shape[:2]) < noise_prob
        noise = np.random.normal(0, 10, img.shape[:2])
        img[noise_mask] = np.clip(img[noise_mask] + noise[noise_mask], 0, 255)
        return img
    
    # 3. Gradient Fade (특정 부분을 점진적으로 흐리게)
    def apply_gradient_fade(img, num_regions=2):
        img = img.copy()
        for _ in range(num_regions):
            x = np.random.randint(0, width - 10)
            y = np.random.randint(0, height - 10)
            w = np.random.randint(5, 10)
            h = np.random.randint(5, 10)
            
            # 그라데이션 마스크 생성
            gradient = np.linspace(1, 0.3, w).reshape(1, -1)
            gradient = np.repeat(gradient, h, axis=0)
            
            region = img[y:y+h, x:x+w] * gradient
            img[y:y+h, x:x+w] = region
        return img
    
    # 4. Random Line Break (가는 선으로 패턴 끊기)
    def apply_line_break(img, num_lines=2):
        img = img.copy()
        for _ in range(num_lines):
            thickness = np.random.randint(1, 3)
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)
            length = np.random.randint(3, 7)
            angle = np.random.randint(0, 180)
            
            x2 = int(x1 + length * np.cos(np.radians(angle)))
            y2 = int(y1 + length * np.sin(np.radians(angle)))
            
            # 주변 픽셀값의 평균으로 선 그리기
            surrounding = img[max(0, y1-2):min(height, y1+3),
                           max(0, x1-2):min(width, x1+3)]
            line_color = int(np.mean(surrounding))
            cv2.line(img, (x1, y1), (x2, y2), line_color, thickness)
        return img
    
    # 각 augmentation 적용
    for _ in range(3):  # 각 방법당 3개의 변형 생성
        # Cutout
        aug_img = apply_cutout(image)
        augmented_images.append(aug_img)
        if mask is not None:
            augmented_masks.append(mask.copy())
        
        # Noise Injection
        aug_img = apply_noise(image)
        augmented_images.append(aug_img)
        if mask is not None:
            augmented_masks.append(mask.copy())
        
        # Gradient Fade
        aug_img = apply_gradient_fade(image)
        augmented_images.append(aug_img)
        if mask is not None:
            augmented_masks.append(mask.copy())
        
        # Line Break
        aug_img = apply_line_break(image)
        augmented_images.append(aug_img)
        if mask is not None:
            augmented_masks.append(mask.copy())
        
        # 조합 적용
        aug_img = apply_line_break(apply_cutout(image))
        augmented_images.append(aug_img)
        if mask is not None:
            augmented_masks.append(mask.copy())
    
    if mask is not None:
        return augmented_images, augmented_masks
    return augmented_images