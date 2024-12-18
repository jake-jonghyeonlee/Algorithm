import cv2
import numpy as np
from pathlib import Path

def enhance_image_for_segmentation(image_path, output_path=None):
    """
    광학계 이미지를 세그멘테이션 학습에 적합하도록 전처리하는 함수
    
    Args:
        image_path (str): 입력 이미지 경로
        output_path (str, optional): 출력 이미지 저장 경로
    
    Returns:
        numpy.ndarray: 전처리된 이미지
    """
    # 이미지 로드
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("이미지를 불러올 수 없습니다.")

    # 컬러 보정
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 경계선 강화
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel_sharpening)

    # 노이즈 제거
    denoised = cv2.fastNlMeansDenoisingColored(sharpened)

    # 결과 저장
    if output_path:
        cv2.imwrite(str(output_path), denoised)

    return denoised

def process_directory(input_dir, output_dir):
    """
    디렉토리 내의 모든 이미지를 일괄 처리하는 함수
    
    Args:
        input_dir (str): 입력 이미지들이 있는 디렉토리 경로
        output_dir (str): 전처리된 이미지를 저장할 디렉토리 경로
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for img_path in input_path.iterdir():
        if img_path.suffix.lower() in image_extensions:
            output_file = output_path / img_path.name
            try:
                enhance_image_for_segmentation(img_path, output_file)
                print(f"처리 완료: {img_path.name}")
            except Exception as e:
                print(f"처리 실패 {img_path.name}: {str(e)}")

if __name__ == "__main__":
    # 사용 예시
    input_directory = "path/to/input/images"
    output_directory = "path/to/output/images"
    process_directory(input_directory, output_directory)
