import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 배치 차원을 유지하면서 나머지를 평탄화
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        """
        alpha: false positive 가중치
        beta: false negative 가중치
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 배치 차원을 유지하면서 나머지를 평탄화
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # True Positive, False Positive, False Negative
        TP = (pred * target).sum(dim=1)
        FP = (pred * (1-target)).sum(dim=1)
        FN = ((1-pred) * target).sum(dim=1)
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        """
        alpha: 균형 가중치
        gamma: 집중 파라미터
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # 배치 차원을 유지하면서 나머지를 평탄화
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Focal term
        pt = target * pred + (1 - target) * (1 - pred)
        focal_term = (1 - pt) ** self.gamma
        
        # 결합
        focal_loss = self.alpha * focal_term * bce
        
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        return self.dice_weight * dice_loss + self.focal_weight * focal_loss

def get_loss_function(loss_type='dice', **kwargs):
    """
    손실 함수를 선택하는 팩토리 함수
    
    Args:
        loss_type: str, ['dice', 'tversky', 'focal', 'combined'] 중 하나
        **kwargs: 각 손실 함수에 대한 추가 인자
    
    Returns:
        선택된 손실 함수 모듈
    """
    loss_functions = {
        'dice': DiceLoss,
        'tversky': TverskyLoss,
        'focal': FocalLoss,
        'combined': CombinedLoss
    }
    
    if loss_type not in loss_functions:
        raise ValueError(f"Loss type '{loss_type}'가 지원되지 않습니다. 다음 중 선택하세요: {list(loss_functions.keys())}")
    
    return loss_functions[loss_type](**kwargs)


import numpy as np
import torch

def calculate_iou(preds, targets, num_classes):
    iou_list = []
    preds = preds.argmax(dim=1)  # 예측된 클래스 인덱스
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        union = ((preds == cls) | (targets == cls)).sum().item()
        iou = intersection / (union + 1e-6)  # 1e-6은 0으로 나누는 것을 방지
        iou_list.append(iou)
    return np.mean(iou_list)

def calculate_dice(preds, targets, num_classes):
    dice_list = []
    preds = preds.argmax(dim=1)  # 예측된 클래스 인덱스
    for cls in range(num_classes):
        intersection = ((preds == cls) & (targets == cls)).sum().item()
        dice = (2 * intersection) / (preds[targets == cls].sum().item() + targets[targets == cls].sum().item() + 1e-6)
        dice_list.append(dice)
    return np.mean(dice_list)

def evaluate_model(model, dataloader, num_classes, device):
    model.eval()
    total_iou = 0
    total_dice = 0
    total_samples = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            total_iou += calculate_iou(outputs, masks, num_classes)
            total_dice += calculate_dice(outputs, masks, num_classes)
            total_samples += 1

    avg_iou = total_iou / total_samples
    avg_dice = total_dice / total_samples
    return avg_iou, avg_dice

