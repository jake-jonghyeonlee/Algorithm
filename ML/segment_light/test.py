import torch
from PIL import Image
import os
from optimize import LightUNet
from torchvision import transforms
import numpy as np

def test_single_image(model_path, image_path, output_dir):
    """
    단일 이미지에 대한 세그멘테이션 예측을 수행하고 결과를 저장
    """
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = LightUNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 예측
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # 예측 결과를 이미지로 변환
    pred_mask = prediction.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    
    # 결과 저장
    output_path = os.path.join(
        output_dir, 
        f"pred_{os.path.basename(image_path).replace('.jpg', '.png')}"
    )
    Image.fromarray(pred_mask).save(output_path)
    print(f'Prediction saved to {output_path}')

def test_directory(model_path, test_dir, output_dir):
    """
    디렉토리 내의 모든 이미지에 대해 세그멘테이션 수행
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미지 파일 리스트 가져오기
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        test_single_image(model_path, image_path, output_dir)

if __name__ == '__main__':
    # 설정
    MODEL_PATH = 'best_model.pkl'
    TEST_DIR = 'data/test/images'  # 테스트 이미지가 있는 디렉토리
    OUTPUT_DIR = 'data/test/predictions'  # 예측 결과를 저장할 디렉토리
    
    test_directory(MODEL_PATH, TEST_DIR, OUTPUT_DIR) 