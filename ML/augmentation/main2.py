def subtle_augmentation(image):
    # 매우 작은 범위의 augmentation을 적용
    augmented_images = []
    
    # 원본 이미지 추가
    augmented_images.append(image)
    
    # 1. 미세한 이동 (-2~2 픽셀)
    for tx in [-2, -1, 1, 2]:
        for ty in [-2, -1, 1, 2]:
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
            augmented_images.append(shifted)
    
    # 2. 작은 각도 회전 (-5~5도)
    for angle in [-5, -2.5, 2.5, 5]:
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        augmented_images.append(rotated)
    
    # 3. 미세한 스케일 변화 (0.95~1.05)
    for scale in [0.95, 0.975, 1.025, 1.05]:
        scaled = cv2.resize(image, None, fx=scale, fy=scale)
        # 원본 크기로 패딩
        if scale < 1:
            pad_h = (image.shape[0] - scaled.shape[0]) // 2
            pad_w = (image.shape[1] - scaled.shape[1]) // 2
            scaled = np.pad(scaled, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
        else:
            scaled = cv2.resize(scaled, (image.shape[1], image.shape[0]))
        augmented_images.append(scaled)
    
    # 4. 밝기와 대비 조절
    for alpha in [0.9, 1.1]:  # 대비
        for beta in [-10, 10]:  # 밝기
            adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            augmented_images.append(adjusted)
    
    # 5. 감마 보정
    for gamma in [0.8, 1.2]:
        gamma_corrected = np.power(image / 255.0, gamma) * 255.0
        gamma_corrected = gamma_corrected.astype(np.uint8)
        augmented_images.append(gamma_corrected)
    
    # 6. 매우 작은 가우시안 블러
    for ksize in [(3,3), (5,5)]:
        blurred = cv2.GaussianBlur(image, ksize, 0)
        augmented_images.append(blurred)
    
    return augmented_images

def train_with_augmentation(model, train_images, train_labels):
    # 각 에포크마다 새로운 augmentation 적용
    augmented_images = []
    augmented_labels = []
    
    for image, label in zip(train_images, train_labels):
        aug_images = subtle_augmentation(image)
        augmented_images.extend(aug_images)
        augmented_labels.extend([label] * len(aug_images))
    
    # 데이터 셔플
    indices = np.random.permutation(len(augmented_images))
    augmented_images = np.array(augmented_images)[indices]
    augmented_labels = np.array(augmented_labels)[indices]
    
    return augmented_images, augmented_labels