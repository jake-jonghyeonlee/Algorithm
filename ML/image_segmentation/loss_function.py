import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Focal loss formula
        pt = target * pred + (1 - target) * (1 - pred)
        w = self.alpha * target + (1 - self.alpha) * (1 - target)
        w = w * (1 - pt).pow(self.gamma)
        
        return F.binary_cross_entropy(pred, target, w, reduction='mean')

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (pred * target).sum()
        FP = ((1-target) * pred).sum()
        FN = (target * (1-pred)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        
        return 1 - Tversky

class ComboLoss(nn.Module):
    def __init__(self, weights=None):
        super(ComboLoss, self).__init__()
        self.weights = weights if weights is not None else {
            'bce': 1.0,
            'dice': 1.0,
            'focal': 1.0
        }
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        
    def forward(self, pred, target):
        loss = 0
        if self.weights['bce'] > 0:
            loss += self.weights['bce'] * self.bce(pred, target)
        if self.weights['dice'] > 0:
            loss += self.weights['dice'] * self.dice(pred, target)
        if self.weights['focal'] > 0:
            loss += self.weights['focal'] * self.focal(pred, target)
        return loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        total = (pred + target).sum()
        union = total - intersection
        
        IoU = (intersection + self.smooth)/(union + self.smooth)
        
        return 1 - IoU

def get_loss_function(loss_params):
    loss_type = loss_params['loss_type']
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_type == 'dice':
        return DiceLoss(smooth=loss_params.get('smooth', 1.0))
    elif loss_type == 'focal':
        return FocalLoss(alpha=loss_params.get('alpha', 0.25),
                        gamma=loss_params.get('gamma', 2))
    elif loss_type == 'tversky':
        return TverskyLoss(alpha=loss_params.get('alpha', 0.3),
                          beta=loss_params.get('beta', 0.7),
                          smooth=loss_params.get('smooth', 1.0))
    elif loss_type == 'iou':
        return IoULoss(smooth=loss_params.get('smooth', 1.0))
    elif loss_type == 'combo':
        weights = {
            'bce': loss_params.get('bce_weight', 1.0),
            'dice': loss_params.get('dice_weight', 1.0),
            'focal': loss_params.get('focal_weight', 1.0)
        }
        return ComboLoss(weights=weights)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}") 