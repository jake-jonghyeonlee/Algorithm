

import cv2
import numpy as np
from typing import Tuple, List

class ImageAnalyzer:
    def __init__(self):
        self.min_thickness = 3  # 최소 두께 기준 (픽셀)
        
    def detect_thin_overlaps(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """두 이미지의 중복 영역 중 얇은 부분을 검출"""
        # 이미지 크기 확인
        if img1.shape != (1024, 1024) or img2.shape != (1024, 1024):
            raise ValueError("이미지는 1024x1024 크기여야 합니다.")
            
        # 이미지 이진화
        _, bin1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, bin2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
        
        # 중복 영역 찾기
        overlap = cv2.bitwise_and(bin1, bin2)
        
        # 거리 변환으로 두께 측정
        dist = cv2.distanceTransform(overlap, cv2.DIST_L2, 5)
        
        # 얇은 부분만 추출
        thin_areas = np.where(dist < self.min_thickness, overlap, 0)
        
        return thin_areas.astype(np.uint8)
    
    def find_nearest_contours(self, img: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """가장 가까운 컨투어 쌍들을 찾기"""
        # 컨투어 검출
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        nearest_pairs = []
        
        # 모든 컨투어 쌍에 대해 최소 거리 계산
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                min_dist = float('inf')
                
                # 각 컨투어의 모든 점에 대해 최소 거리 계산
                for pt1 in contours[i]:
                    for pt2 in contours[j]:
                        dist = np.linalg.norm(pt1[0] - pt2[0])
                        min_dist = min(min_dist, dist)
                
                nearest_pairs.append((contours[i], contours[j], min_dist))
        
        # 거리순으로 정렬
        nearest_pairs.sort(key=lambda x: x[2])
        
        return nearest_pairs

    def visualize_results(self, img: np.ndarray, nearest_pairs: List[Tuple[np.ndarray, np.ndarray, float]], 
                         num_pairs: int = 5) -> np.ndarray:
        """결과 시각화"""
        result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 가장 가까운 N개의 컨투어 쌍 시각화
        for i, (cont1, cont2, dist) in enumerate(nearest_pairs[:num_pairs]):
            color = (0, 255 - i*40, i*40)  # 각 쌍마다 다른 색상
            cv2.drawContours(result, [cont1], -1, color, 2)
            cv2.drawContours(result, [cont2], -1, color, 2)
            
            # 가장 가까운 점들을 연결하는 선 그리기
            m1 = np.mean(cont1, axis=0)[0].astype(np.int32)
            m2 = np.mean(cont2, axis=0)[0].astype(np.int32)
            cv2.line(result, tuple(m1), tuple(m2), color, 1)
            
        return result