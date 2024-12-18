import cv2
import numpy as np
import os

# 템플릿 매칭 방법 정의
TEMPLATE_MATCHING_METHODS = {
    'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,  # 정규화된 상관계수 방법: -1에서 1사이 값, 1에 가까울수록 유사
    'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,    # 정규화된 상관관계 방법: 0에서 1사이 값, 1에 가까울수록 유사
    'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED,  # 정규화된 제곱차 방법: 0에서 1사이 값, 0에 가까울수록 유사
    'TM_CCOEFF': cv2.TM_CCOEFF,                # 상관계수 방법: 완벽한 매칭일 때 큰 양수값
    'TM_CCORR': cv2.TM_CCORR,                  # 상관관계 방법: 완벽한 매칭일 때 큰 값
    'TM_SQDIFF': cv2.TM_SQDIFF                 # 제곱차 방법: 완벽한 매칭일 때 0에 가까운 값
}

def template_matching(reference_path, target_path, output_path, method='TM_CCOEFF_NORMED', threshold=0.8):
    """
    템플릿 매칭을 수행하여 유사한 영역을 크롭하는 함수
    
    Args:
        reference_path (str): 레퍼런스 이미지 경로
        target_path (str): 대상 이미지 경로
        output_path (str): 결과 이미지 저장 경로
        method (str): 템플릿 매칭 방법
        threshold (float): 매칭 임계값 (0~1)
    """
    # 이미지 로드
    reference = cv2.imread(reference_path)
    target = cv2.imread(target_path)
    
    # 그레이스케일 변환
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # 템플릿 매칭 수행
    matching_method = TEMPLATE_MATCHING_METHODS[method]
    result = cv2.matchTemplate(target_gray, reference_gray, matching_method)
    
    # 최대/최소 매칭 위치 찾기
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 매칭 방법에 따라 최적 위치와 점수 결정
    if method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED']:
        score = 1 - min_val if method == 'TM_SQDIFF_NORMED' else min_val
        top_left = min_loc
    else:
        score = max_val
        top_left = max_loc
    
    # 임계값 체크
    if (method in ['TM_SQDIFF', 'TM_SQDIFF_NORMED'] and score <= threshold) or \
       (method not in ['TM_SQDIFF', 'TM_SQDIFF_NORMED'] and score >= threshold):
        
        # 레퍼런스 이미지 크기
        h, w = reference_gray.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # 매칭된 영역 표시 및 크롭
        result_img = target.copy()
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
        
        # 매칭 스코어 표시
        score_text = f'Score: {score:.3f}'
        cv2.putText(result_img, score_text, 
                   (top_left[0], top_left[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 크롭된 이미지
        cropped = target[top_left[1]:bottom_right[1], 
                        top_left[0]:bottom_right[0]]
        
        # 결과 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 원본 이미지에 매칭 결과 표시한 이미지 저장
        result_path = output_path.replace('.', '_matched.')
        cv2.imwrite(result_path, result_img)
        
        # 크롭된 이미지 저장
        cv2.imwrite(output_path, cropped)
        
        return True, score
    
    return False, score

def process_multiple_images(reference_path, target_dir, output_dir, method='TM_CCOEFF_NORMED', threshold=0.8):
    """
    여러 이미지에 대해 템플릿 매칭 수행
    
    Args:
        reference_path (str): 레퍼런스 이미지 경로
        target_dir (str): 대상 이미지들이 있는 디렉토리
        output_dir (str): 결과 저장 디렉토리
        method (str): 템플릿 매칭 방법
        threshold (float): 매칭 임계값
    """
    if method not in TEMPLATE_MATCHING_METHODS:
        raise ValueError(f"지원하지 않는 매칭 방법입니다. 사용 가능한 방법: {list(TEMPLATE_MATCHING_METHODS.keys())}")
    
    for filename in os.listdir(target_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            target_path = os.path.join(target_dir, filename)
            output_path = os.path.join(output_dir, f'cropped_{filename}')
            
            success, score = template_matching(
                reference_path, 
                target_path, 
                output_path, 
                method,
                threshold
            )
            
            if success:
                print(f'Successfully processed {filename} (score: {score:.3f})')
            else:
                print(f'Failed to find match for {filename} (score: {score:.3f})')

if __name__ == "__main__":
    # 사용 예시
    reference_path = "path/to/reference.jpg"
    target_dir = "path/to/target/directory"
    output_dir = "path/to/output/directory"
    
    # 템플릿 매칭 방법 선택 (기본값: TM_CCOEFF_NORMED)
    matching_method = 'TM_CCOEFF_NORMED'
    
    process_multiple_images(
        reference_path, 
        target_dir, 
        output_dir, 
        method=matching_method,
        threshold=0.8
    )
