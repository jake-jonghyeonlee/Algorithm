import torch
import cv2
import numpy as np
from pathlib import Path
from models import UNet, AttentionUNet, ResUNet, DenseUNet, SEUNet, DeepLabV3Plus
from augmentation import get_transforms
import matplotlib.pyplot as plt

def load_model(checkpoint_path):
    # 체크포인트 로드
    checkpoint = torch.load(checkpoint_path)
    
    # 모델 타입에 따른 모델 생성
    model_type = checkpoint['model_type']
    if model_type == "UNet":
        model = UNet()
    elif model_type == "AttentionUNet":
        model = AttentionUNet()
    elif model_type == "ResUNet":
        model = ResUNet()
    elif model_type == "DenseUNet":
        model = DenseUNet()
    elif model_type == "DeepLabV3Plus":
        model = DeepLabV3Plus()
    else:
        model = SEUNet()
    
    # 모델 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint['transform_params']

def predict_mask(model, image, transform):
    # 이미지 전처리
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    
    # Transform 적용
    augmented = transform(image=image)
    image = augmented['image']
    
    # 배치 차원 추가
    image = image.unsqueeze(0)
    
    # 예측
    model.eval()
    with torch.no_grad():
        pred = model(image)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
    
    # 배치 차원 제거 및 CPU로 이동
    pred = pred.squeeze().cpu().numpy()
    return pred

def visualize_results(image, mask, save_path):
    plt.figure(figsize=(15, 5))
    
    # 원본 이미지
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    # 예측 마스크
    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # 오버레이
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    overlay[mask == 1] = [255, 0, 0]  # 빨간색으로 마스크 표시
    
    plt.subplot(133)
    plt.imshow(overlay)
    plt.title('Overlay')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 경로 설정
    model_path = Path("models/best_model.pkl")
    test_dir = Path("data/images/test")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform_params = load_model(model_path)
    model = model.to(device)
    
    # Transform 설정
    transform = get_transforms(train=False, aug_params=transform_params)
    
    # 테스트 이미지 처리
    for img_path in test_dir.glob('*.png'):  # 또는 *.jpg
        # 이미지 로드
        image = cv2.imread(str(img_path))
        
        # 마스크 예측
        mask = predict_mask(model, image, transform)
        
        # 결과 저장
        output_path = output_dir / f"{img_path.stem}_result.png"
        visualize_results(image, mask, output_path)
        
        # 마스크 파일 따로 저장
        mask_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        
        print(f"Processed: {img_path.name}")

if __name__ == "__main__":
    main() 

    '''
    data/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
└── masks/
    ├── train/
    └── valid/

models/
└── best_model.pkl

results/
├── image1_result.png
├── image1_mask.png
├── image2_result.png
└── image2_mask.png'''