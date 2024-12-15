

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
    
    def find_close_contours(self, img: np.ndarray) -> List[Dict]:
        """컨투어 간의 관계를 분석하여 연관된 컨투어 쌍 검출
        
        컨투어 간의 관계는 다음 3가지 특성으로 분석됩니다:
        1. 방향 유사도: 컨투어의 주방향 벡터 간의 각도 차이
           - cv2.fitEllipse()로 타원 근사하여 장축 방향 계산
           - 각도가 비슷할수록 높은 점수 (0-1)
           
        2. 크기 유사도: 컨투어 면적 비율
           - cv2.contourArea()로 계산한 면적 비교
           - 면적이 비슷할수록 높은 점수 (0-1)
           
        3. 거리 점수: 중심점 간 거리의 지수 감쇠
           - 거리가 가까울수록 높은 점수 (0-1)
           - exp(-d/100)으로 계산하여 거리 증가에 따라 자연스럽게 감소
        
        최종 관계 점수는 위 3가지 점수의 평균으로 계산됩니다.
        점수가 0.3 이상인 쌍만 연관된 것으로 판단합니다.
        
        Args:
            img: 입력 이미지 (바이너리)
            
        Returns:
            연관된 컨투어 쌍들의 정보를 담은 리스트
            [{'contour1': cont1, 'contour2': cont2, 'distance': dist, 
              'relation_score': score, 'direction_similarity': dir_sim,
              'size_similarity': size_sim, 'distance_score': dist_score}, ...]
        """
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        related_pairs = []
        
        for i in range(len(contours)):
            for j in range(i + 1, len(contours)):
                # 1. 방향 유사도 계산
                _, _, angle1 = cv2.fitEllipse(contours[i]) if len(contours[i]) > 4 else (0, 0, 0)
                _, _, angle2 = cv2.fitEllipse(contours[j]) if len(contours[j]) > 4 else (0, 0, 0)
                angle_diff = abs(angle1 - angle2) % 180
                direction_similarity = 1 - (angle_diff / 180)
                
                # 2. 크기 유사도 계산
                area1 = cv2.contourArea(contours[i])
                area2 = cv2.contourArea(contours[j])
                size_similarity = min(area1, area2) / max(area1, area2)
                
                # 3. 거리 점수 계산
                m1 = np.mean(contours[i], axis=0)[0]
                m2 = np.mean(contours[j], axis=0)[0]
                center_dist = np.linalg.norm(m1 - m2)
                distance_score = np.exp(-center_dist / 100)
                
                # 최종 관계 점수 계산
                relation_score = (direction_similarity + size_similarity + distance_score) / 3
                
                if relation_score > 0.3:
                    related_pairs.append({
                        'contour1': contours[i],
                        'contour2': contours[j],
                        'distance': center_dist,
                        'relation_score': relation_score,
                        'direction_similarity': direction_similarity,
                        'size_similarity': size_similarity, 
                        'distance_score': distance_score
                    })
        
        related_pairs.sort(key=lambda x: x['relation_score'], reverse=True)
        return related_pairs