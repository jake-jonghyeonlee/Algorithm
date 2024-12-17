import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return torch.mean(focal_loss)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return 1 - tversky

class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        """
        weights: 각 loss function의 가중치 딕셔너리
        예: {'bce': 1.0, 'focal': 1.0, 'dice': 1.0, 'tversky': 1.0}
        """
        super().__init__()
        self.weights = weights or {
            'bce': 1.0,
            'focal': 1.0,
            'dice': 1.0,
            'tversky': 1.0
        }
        
        self.bce = nn.BCELoss()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.tversky = TverskyLoss()
    
    def forward(self, inputs, targets):
        loss = 0
        if self.weights['bce'] > 0:
            loss += self.weights['bce'] * self.bce(inputs, targets)
        if self.weights['focal'] > 0:
            loss += self.weights['focal'] * self.focal(inputs, targets)
        if self.weights['dice'] > 0:
            loss += self.weights['dice'] * self.dice(inputs, targets)
        if self.weights['tversky'] > 0:
            loss += self.weights['tversky'] * self.tversky(inputs, targets)
        return loss

def get_loss_function(loss_type='combo', **kwargs):
    """
    손실 함수를 반환하는 헬퍼 함수
    Args:
        loss_type: 'bce', 'focal', 'dice', 'tversky', 'combo' 중 하나
        **kwargs: 각 손실 함수의 파라미터
    Returns:
        손실 함수 인스턴스
    """
    if loss_type == 'bce':
        return nn.BCELoss()
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    elif loss_type == 'tversky':
        return TverskyLoss(**kwargs)
    elif loss_type == 'combo':
        return ComboLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 