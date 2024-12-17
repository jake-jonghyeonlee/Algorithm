import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score

class SegmentationMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
    
    def _prepare_inputs(self, y_pred, y_true):
        # 예측값과 실제값을 numpy 배열로 변환
        if torch.is_tensor(y_pred):
            y_pred = y_pred.cpu().detach().numpy()
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().detach().numpy()
            
        # 이진화
        y_pred = (y_pred > self.threshold).astype(np.int64)
        y_true = (y_true > self.threshold).astype(np.int64)
        
        # 1차원 배열로 변환
        y_pred = y_pred.ravel()
        y_true = y_true.ravel()
        
        return y_pred, y_true
    
    def accuracy(self, y_pred, y_true):
        y_pred, y_true = self._prepare_inputs(y_pred, y_true)
        return accuracy_score(y_true, y_pred)
    
    def f1(self, y_pred, y_true):
        y_pred, y_true = self._prepare_inputs(y_pred, y_true)
        return f1_score(y_true, y_pred)
    
    def dice_coefficient(self, y_pred, y_true):
        y_pred, y_true = self._prepare_inputs(y_pred, y_true)
        intersection = np.sum(y_pred * y_true)
        return (2. * intersection) / (np.sum(y_pred) + np.sum(y_true))
    
    def jaccard(self, y_pred, y_true):
        y_pred, y_true = self._prepare_inputs(y_pred, y_true)
        return jaccard_score(y_true, y_pred)
    
    def get_metrics(self, y_pred, y_true):
        """
        모든 메트릭을 한 번에 계산하여 딕셔너리로 반환
        """
        return {
            'Accuracy': self.accuracy(y_pred, y_true),
            'F1 Score': self.f1(y_pred, y_true),
            'Dice Coefficient': self.dice_coefficient(y_pred, y_true),
            'Jaccard Score': self.jaccard(y_pred, y_true)
        }

def print_metrics(metrics_dict, prefix=""):
    """
    메트릭 결과를 보기 좋게 출력
    """
    print(f"\n{prefix} Metrics:")
    print("-" * 50)
    for metric_name, value in metrics_dict.items():
        print(f"{metric_name}: {value:.4f}")
    print("-" * 50) 