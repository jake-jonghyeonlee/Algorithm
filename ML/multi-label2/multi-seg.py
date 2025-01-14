# 평가 데이터 로더 설정
from torch.utils.data import DataLoader

# 예를 들어, test_dataset을 설정한 후
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 모델 평가
avg_iou, avg_dice = evaluate_model(model, test_loader, num_classes, device)
print(f'Average IoU: {avg_iou:.4f}, Average Dice: {avg_dice:.4f}')

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

import torch
import torch.nn as nn

class MultiClassCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-1, reduction='mean'):
        """
        Args:
            weight (Tensor, optional): 각 클래스에 대한 가중치. 기본값은 None입니다.
            ignore_index (int, optional): 무시할 클래스 인덱스. 기본값은 -1입니다.
            reduction (string, optional): 'none', 'mean', 'sum' 중 하나. 기본값은 'mean'입니다.
        """
        super(MultiClassCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 모델의 출력 (N, C, H, W) 형태.
            targets (Tensor): 정답 마스크 (N, H, W) 형태.

        Returns:
            Tensor: 손실 값.
        """
        # CrossEntropyLoss는 (N, C, H, W) 형태의 출력을 요구하므로, 
        # preds는 (N, H, W) 형태로 변환되어야 합니다.
        return self.criterion(outputs, targets)

# 사용 예시
if __name__ == "__main__":
    # 임의의 모델 출력 및 타겟 생성
    outputs = torch.randn(2, 4, 5, 5)  # (배치 크기, 클래스 수, 높이, 너비)
    targets = torch.randint(0, 4, (2, 5, 5))  # (배치 크기, 높이, 너비)

    # 손실 함수 초기화
    loss_fn = MultiClassCrossEntropyLoss()

    # 손실 계산
    loss = loss_fn(outputs, targets)
    print(f"Calculated Loss: {loss.item()}")

import torch
import torch.nn as nn

class MultiClassTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Args:
            alpha (float): False Positive 가중치
            beta (float): False Negative 가중치
            smooth (float): 0으로 나누는 것을 방지하기 위한 작은 값
        """
        super(MultiClassTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 모델의 출력 (N, C, H, W) 형태.
            targets (Tensor): 정답 마스크 (N, H, W) 형태.

        Returns:
            Tensor: Tversky 손실 값.
        """
        # 예측된 클래스의 인덱스
        preds = outputs.argmax(dim=1)  # (N, H, W) 형태로 변환
        num_classes = outputs.shape[1]  # 클래스 수

        tversky_list = []
        for cls in range(num_classes):
            # 클래스별 마스크 생성
            true_cls = (targets == cls).float()  # 정답 마스크
            pred_cls = (preds == cls).float()  # 예측 마스크

            # True Positive, False Positive, False Negative 계산
            TP = (true_cls * pred_cls).sum()
            FP = (pred_cls * (1 - true_cls)).sum()
            FN = ((1 - pred_cls) * true_cls).sum()

            # Tversky 계수 계산
            tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
            tversky_list.append(tversky)

        # 평균 Tversky 손실 계산
        return 1 - torch.mean(torch.tensor(tversky_list))

# 사용 예시
if __name__ == "__main__":
    # 임의의 모델 출력 및 타겟 생성
    outputs = torch.randn(2, 4, 5, 5)  # (배치 크기, 클래스 수, 높이, 너비)
    targets = torch.randint(0, 4, (2, 5, 5))  # (배치 크기, 높이, 너비)

    # 손실 함수 초기화
    loss_fn = MultiClassTverskyLoss()

    # 손실 계산
    loss = loss_fn(outputs, targets)
    print(f"Calculated Tversky Loss: {loss.item()}")

    import torch
import torch.nn as nn

class MultiClassDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Args:
            smooth (float): 0으로 나누는 것을 방지하기 위한 작은 값
        """
        super(MultiClassDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 모델의 출력 (N, C, H, W) 형태.
            targets (Tensor): 정답 마스크 (N, H, W) 형태.

        Returns:
            Tensor: Dice 손실 값.
        """
        preds = outputs.argmax(dim=1)  # (N, H, W) 형태로 변환
        num_classes = outputs.shape[1]  # 클래스 수

        dice_list = []
        for cls in range(num_classes):
            # 클래스별 마스크 생성
            true_cls = (targets == cls).float()  # 정답 마스크
            pred_cls = (preds == cls).float()  # 예측 마스크

            # True Positive 계산
            intersection = (true_cls * pred_cls).sum()

            # Dice 계수 계산
            dice = (2. * intersection + self.smooth) / (true_cls.sum() + pred_cls.sum() + self.smooth)
            dice_list.append(dice)

        # 평균 Dice 손실 계산
        return 1 - torch.mean(torch.tensor(dice_list))

# 사용 예시
if __name__ == "__main__":
    # 임의의 모델 출력 및 타겟 생성
    outputs = torch.randn(2, 4, 5, 5)  # (배치 크기, 클래스 수, 높이, 너비)
    targets = torch.randint(0, 4, (2, 5, 5))  # (배치 크기, 높이, 너비)

    # 손실 함수 초기화
    loss_fn = MultiClassDiceLoss()

    # 손실 계산
    loss = loss_fn(outputs, targets)
    print(f"Calculated Dice Loss: {loss.item()}")

import torch
import torch.optim as optim

# 모델, 손실 함수, 옵티마이저 초기화
model = ...  # 모델 정의
criterion = MultiClassDiceLoss()  # 또는 MultiClassTverskyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(num_epochs):
    for images, targets in dataloader:
        # 1. 모델의 출력 계산
        outputs = model(images)

        # 2. 손실 계산
        loss = criterion(outputs, targets)

        # 3. 기울기 초기화
        optimizer.zero_grad()

        # 4. 손실에 대한 기울기 계산
        loss.backward()

        # 5. 가중치 업데이트
        optimizer.step()

        # 6. 손실 값 출력
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")