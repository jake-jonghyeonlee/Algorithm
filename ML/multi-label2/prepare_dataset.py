import os
import shutil
from PIL import Image
import numpy as np

def create_dataset_structure(base_path):
    """데이터셋 디렉토리 구조 생성"""
    os.makedirs(os.path.join(base_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_path, 'masks'), exist_ok=True)

def prepare_mask(mask_paths, output_path, num_classes):
    """여러 개의 이진 마스크를 하나의 멀티클래스 마스크로 결합"""
    # 첫 번째 마스크의 크기로 초기화
    base_mask = Image.open(mask_paths[0])
    combined_mask = np.zeros(base_mask.size[::-1], dtype=np.uint8)
    
    # 각 클래스별 마스크를 결합
    for class_idx, mask_path in enumerate(mask_paths, start=1):
        if class_idx >= num_classes:
            break
            
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        
        # 이진 마스크인 경우 (0과 255로 구성)
        if set(np.unique(mask_array)).issubset({0, 255}):
            mask_array = (mask_array > 0).astype(np.uint8)
        
        # 해당 클래스 값으로 설정
        combined_mask[mask_array > 0] = class_idx
    
    # 결과 저장
    result_mask = Image.fromarray(combined_mask)
    result_mask.save(output_path)
    return output_path

def verify_dataset(image_path, mask_path, num_classes):
    """데이터셋 유효성 검증"""
    # 이미지 확인
    img = Image.open(image_path)
    mask = Image.open(mask_path)
    
    # 크기가 동일한지 확인
    assert img.size == mask.size, f"Size mismatch: {image_path} ({img.size}) vs {mask_path} ({mask.size})"
    
    # 마스크 값이 올바른 범위인지 확인
    mask_array = np.array(mask)
    unique_values = np.unique(mask_array)
    assert all(val < num_classes for val in unique_values), \
        f"Invalid mask values in {mask_path}. Found values: {unique_values}"

def prepare_dataset_example():
    """데이터셋 준비 예제"""
    base_path = "dataset"
    num_classes = 4
    
    # 디렉토리 구조 생성
    create_dataset_structure(base_path)
    
    # 예제: 단일 이미지에 대한 마스크 준비
    image_name = "example"
    mask_paths = [
        f"raw_masks/{image_name}_class1.png",
        f"raw_masks/{image_name}_class2.png",
        f"raw_masks/{image_name}_class3.png"
    ]
    
    # 이미지와 마스크 복사/변환
    shutil.copy(
        f"raw_images/{image_name}.jpg",
        os.path.join(base_path, 'images', f"{image_name}.jpg")
    )
    
    prepare_mask(
        mask_paths,
        os.path.join(base_path, 'masks', f"{image_name}_mask.png"),
        num_classes
    )

def get_dataset_paths(base_path):
    """데이터셋 경로 리스트 생성"""
    image_dir = os.path.join(base_path, 'images')
    mask_dir = os.path.join(base_path, 'masks')
    
    image_paths = []
    mask_paths = []
    
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, img_name)
            mask_name = img_name.rsplit('.', 1)[0] + '_mask.png'
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
    
    return image_paths, mask_paths

if __name__ == "__main__":
    # 예제 실행
    prepare_dataset_example()
    
    # 데이터셋 경로 가져오기
    base_path = "dataset"
    image_paths, mask_paths = get_dataset_paths(base_path)
    
    # 데이터셋 검증
    num_classes = 4
    for img_path, mask_path in zip(image_paths, mask_paths):
        verify_dataset(img_path, mask_path, num_classes)
    
    print(f"처리된 이미지 수: {len(image_paths)}")
    print("데이터셋 준비가 완료되었습니다!")
