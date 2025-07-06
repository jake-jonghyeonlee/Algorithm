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


import numpy as np
import torch

def calculate_class_weights(dataset):
    """
    데이터셋에서 클래스별 픽셀 점유율을 기반으로 클래스 가중치를 계산합니다.
    
    Args:
        dataset: SegmentationDataset 인스턴스
    
    Returns:
        class_weights: 각 클래스에 대한 가중치
    """
    total_pixels = 0
    class_counts = np.zeros(dataset.num_classes)

    for img, mask in dataset:
        total_pixels += mask.numel()  # 전체 픽셀 수
        for cls in range(dataset.num_classes):
            class_counts[cls] += (mask == cls).sum().item()  # 각 클래스의 픽셀 수

    # 클래스별 점유율 계산
    class_weights = total_pixels / (len(dataset) * class_counts + 1e-6)  # 1e-6은 0으로 나누는 것을 방지
    class_weights = class_weights / np.sum(class_weights)  # 정규화

    return torch.FloatTensor(class_weights)

# 사용 예시
if __name__ == "__main__":
    # 데이터셋 초기화
    dataset = SegmentationDataset(image_paths, mask_paths, num_classes=4)
    
    # 클래스 가중치 계산
    class_weights = calculate_class_weights(dataset)
    print("Calculated Class Weights:", class_weights)

criterion = MultiClassCrossEntropyLoss(weight=class_weights)
'''
멀티 클래스 세그멘테이션에서 배경(Background) 클래스가 넓은 영역을 차지할 경우, 학습에 여러 가지 방식으로 영향을 미칠 수 있습니다. 다음은 그 영향과 배경을 손실에서 제거하는 것의 장단점에 대한 설명입니다.

배경 클래스의 영향
클래스 불균형:
배경 클래스가 넓은 영역을 차지하면, 모델이 배경을 쉽게 예측하게 되어 다른 클래스에 대한 예측 성능이 저하될 수 있습니다. 이는 모델이 배경에 대한 예측을 우선시하게 만들고, 다른 클래스에 대한 학습을 소홀히 하게 됩니다.
손실 함수의 편향:
손실 함수가 배경 클래스의 예측을 많이 포함하게 되면, 손실 값이 배경 클래스에 의해 지배받게 됩니다. 이로 인해 모델이 배경 클래스에 대한 예측을 잘하게 되지만, 다른 클래스에 대한 예측 성능이 떨어질 수 있습니다.
과적합:
모델이 배경 클래스에 과도하게 적합하게 되면, 실제로 관심 있는 객체나 클래스에 대해 일반화 성능이 낮아질 수 있습니다.
배경 클래스 제거
배경 클래스를 손실에서 제거하는 것은 다음과 같은 장단점이 있습니다:

장점
모델의 집중력 향상: 배경 클래스를 제거하면 모델이 다른 클래스에 더 집중하게 되어, 관심 있는 객체에 대한 예측 성능이 향상될 수 있습니다.
손실 함수의 균형: 클래스 간의 손실이 균형을 이루게 되어, 모델이 모든 클래스에 대해 더 잘 학습할 수 있습니다.
단점
모델의 일반화 능력 저하: 배경을 제거하면 모델이 배경에 대한 정보를 잃게 되어, 실제 환경에서 배경이 포함된 이미지에 대해 잘 작동하지 않을 수 있습니다.
복잡한 배경 처리: 실제 이미지에서는 배경이 복잡하게 변할 수 있으며, 이를 무시하면 모델의 성능이 저하될 수 있습니다.
대안
가중치 조정: 배경 클래스에 낮은 가중치를 부여하여 손실에서의 영향을 줄일 수 있습니다. 이렇게 하면 배경이 손실에 미치는 영향이 줄어들면서도 배경 정보를 유지할 수 있습니다.
Focal Loss 사용: Focal Loss는 어려운 샘플에 더 많은 가중치를 부여하여 학습을 촉진합니다. 이를 통해 배경 클래스의 영향을 줄일 수 있습니다.
데이터 증강: 배경을 다양하게 변형하여 모델이 다양한 배경을 학습할 수 있도록 합니다.
클래스별 손실 함수: 각 클래스에 대해 별도의 손실 함수를 사용하여 배경과 다른 클래스의 손실을 분리할 수 있습니다.
결론적으로, 배경 클래스를 손실에서 제거하는 것은 특정 상황에서 유용할 수 있지만, 모델의 일반화 능력에 영향을 미칠 수 있습니다. 따라서, 가중치 조정이나 Focal Loss와 같은 대안을 고려하는 것이 좋습니다. 추가적인 질문이나 도움이 필요하시다면 말씀해 주세요!
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): 클래스 가중치
            gamma (float): 조정 파라미터
            reduction (string): 'none', 'mean', 'sum' 중 하나
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 모델의 출력 (N, C, H, W) 형태.
            targets (Tensor): 정답 마스크 (N, H, W) 형태.

        Returns:
            Tensor: Focal Loss 값.
        """
        # Softmax를 사용하여 클래스 확률 계산
        preds = F.softmax(outputs, dim=1)  # (N, C, H, W) 형태
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()  # (N, H, W, C) -> (N, C, H, W)

        # Focal Loss 계산
        BCE_loss = F.binary_cross_entropy(preds, targets_one_hot, reduction='none')  # (N, C, H, W)
        focal_loss = self.alpha * (1 - preds) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 사용 예시
if __name__ == "__main__":
    outputs = torch.randn(2, 4, 5, 5)  # (배치 크기, 클래스 수, 높이, 너비)
    targets = torch.randint(0, 4, (2, 5, 5))  # (배치 크기, 높이, 너비)

    loss_fn = FocalLoss()
    loss = loss_fn(outputs, targets)
    print(f"Calculated Focal Loss: {loss.item()}")

class ClassWiseLoss(nn.Module):
    def __init__(self, criterion):
        """
        Args:
            criterion: 기본 손실 함수 (예: CrossEntropyLoss 또는 FocalLoss)
        """
        super(ClassWiseLoss, self).__init__()
        self.criterion = criterion

    def forward(self, outputs, targets):
        """
        Args:
            outputs (Tensor): 모델의 출력 (N, C, H, W) 형태.
            targets (Tensor): 정답 마스크 (N, H, W) 형태.

        Returns:
            Tensor: 클래스별 손실 값의 합.
        """
        preds = outputs.argmax(dim=1)  # (N, H, W) 형태로 변환
        num_classes = outputs.shape[1]  # 클래스 수

        class_losses = []
        for cls in range(num_classes):
            cls_mask = (targets == cls).float()  # 현재 클래스의 마스크
            if cls_mask.sum() > 0:  # 현재 클래스가 존재하는 경우
                class_loss = self.criterion(outputs, targets) * cls_mask  # 현재 클래스에 대한 손실
                class_losses.append(class_loss.sum() / (cls_mask.sum() + 1e-6))  # 평균 손실

        return torch.mean(torch.tensor(class_losses))

# 사용 예시
if __name__ == "__main__":
    outputs = torch.randn(2, 4, 5, 5)  # (배치 크기, 클래스 수, 높이, 너비)
    targets = torch.randint(0, 4, (2, 5, 5))  # (배치 크기, 높이, 너비)

    base_loss_fn = nn.CrossEntropyLoss()  # 또는 FocalLoss
    class_wise_loss_fn = ClassWiseLoss(criterion=base_loss_fn)
    loss = class_wise_loss_fn(outputs, targets)
    print(f"Calculated Class-wise Loss: {loss.item()}")

# 각 클래스에 대해 모델 정의 및 학습
models = {}
for cls in range(num_classes):
    models[cls] = MySegmentationModel()  # 모델 정의
    # 모델 학습 코드 (데이터 로더, 손실 함수 등)

# 세그멘테이션 결과 결합
def combine_segmentation_results(models, images):
    combined_results = []
    for img in images:
        class_predictions = []
        for cls, model in models.items():
            output = model(img.unsqueeze(0))  # 배치 차원 추가
            preds = output.argmax(dim=1)  # (H, W) 형태
            class_predictions.append(preds)

        # 클래스별 예측 결과를 결합하여 최종 마스크 생성
        final_mask = torch.stack(class_predictions).argmax(dim=0)  # (H, W) 형태
        combined_results.append(final_mask)

    return combined_results

# 사용 예시
images = [...]  # 입력 이미지 리스트
final_segmentations = combine_segmentation_results(models, images)

'''
One-hot 인코딩은 범주형 데이터를 이진 벡터로 변환하는 방법입니다. 멀티 클래스 세그멘테이션 문제에서 주로 사용되며, 각 클래스에 대해 해당 클래스가 존재하는 경우 1, 그렇지 않은 경우 0으로 표현됩니다.

One-hot 인코딩의 원리
클래스 정의:
예를 들어, 4개의 클래스가 있다고 가정합니다: 배경(0), 클래스 A(1), 클래스 B(2), 클래스 C(3).
인코딩:
각 클래스는 다음과 같이 표현됩니다:
배경: [1, 0, 0, 0]
클래스 A: [0, 1, 0, 0]
클래스 B: [0, 0, 1, 0]
클래스 C: [0, 0, 0, 1]
멀티 클래스 세그멘테이션에서의 사용:
이미지의 각 픽셀에 대해 해당 픽셀이 속하는 클래스의 인덱스를 기반으로 one-hot 벡터를 생성합니다.
예를 들어, 특정 픽셀이 클래스 B에 속한다면, 그 픽셀의 one-hot 인코딩은 [0, 0, 1, 0]이 됩니다.
Focal Loss에서의 One-hot 인코딩
Focal Loss 코드에서 one-hot 인코딩을 사용하는 이유는 다음과 같습니다:

다중 클래스 확률 계산:
모델의 출력은 각 클래스에 대한 확률 분포를 나타내며, softmax 함수를 통해 계산됩니다.
이 확률 분포와 one-hot 인코딩된 타겟을 비교하여 손실을 계산합니다.
Binary Cross-Entropy와의 결합:
Focal Loss는 각 클래스에 대해 Binary Cross-Entropy 손실을 계산합니다. 이때, one-hot 인코딩된 타겟을 사용하여 각 클래스에 대한 손실을 독립적으로 계산합니다.
One-hot 인코딩의 구현 예시
아래는 PyTorch에서 one-hot 인코딩을 사용하는 방법의 예시입니다:

import torch

# 예시: 4개의 클래스
num_classes = 4
targets = torch.tensor([0, 1, 2, 3])  # 각 픽셀의 클래스 인덱스

# One-hot 인코딩
targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # (N, C) 형태
print(targets_one_hot)
결과
위 코드를 실행하면 다음과 같은 one-hot 인코딩 결과가 출력됩니다:


tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.]])
이렇게 생성된 one-hot 인코딩은 Focal Loss 계산에 사용되며, 각 클래스에 대한 손실을 독립적으로 평가할 수 있게 해줍니다.

요약
One-hot 인코딩은 각 클래스의 존재 여부를 이진 벡터로 표현하는 방법입니다.
멀티 클래스 세그멘테이션에서 각 픽셀의 클래스 정보를 효과적으로 전달하며, Focal Loss와 같은 손실 함수에서 사용됩니다.
PyTorch에서는 F.one_hot 함수를 사용하여 쉽게 구현할 수 있습니다.
추가적인 질문이나 도움이 필요하시다면 말씀해 주세요!

'''

'''
Cross-Entropy Loss와 One-hot 인코딩
Cross-Entropy Loss:
Cross-Entropy Loss는 모델의 출력 확률 분포와 실제 클래스 간의 차이를 측정합니다.
일반적으로 nn.CrossEntropyLoss는 모델의 출력에 softmax를 적용하여 확률 분포를 계산하고, 정답 클래스 인덱스를 사용하여 손실을 계산합니다.
One-hot 인코딩 사용:
Cross-Entropy Loss를 사용할 때, 정답 클래스가 one-hot 인코딩된 형태로 제공될 수 있습니다.
이 경우, 손실 함수는 각 클래스에 대한 확률을 비교하여 손실을 계산합니다.

import torch
import torch.nn as nn
import torch.nn.functional as F

# 예시: 4개의 클래스
num_classes = 4
targets = torch.tensor([0, 1, 2, 3])  # 각 픽셀의 클래스 인덱스

# One-hot 인코딩
targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # (N, C) 형태

# 모델의 출력 (N, C) 형태로 가정
outputs = torch.randn(4, num_classes)  # (N, C) 형태

# Cross-Entropy Loss 계산
loss_fn = nn.BCELoss()  # Binary Cross-Entropy Loss 사용
loss = loss_fn(outputs.sigmoid(), targets_one_hot)  # Sigmoid를 통해 확률로 변환

print(f"Calculated Loss: {loss.item()}")

주의 사항
nn.CrossEntropyLoss를 사용할 경우, 모델의 출력은 raw logits 형태여야 하며, 이 함수는 내부적으로 softmax를 적용합니다. 따라서, one-hot 인코딩을 사용할 때는 nn.BCELoss와 같은 손실 함수를 사용해야 합니다.
Cross-Entropy Loss를 사용할 때는 일반적으로 클래스 인덱스를 직접 사용하는 것이 더 효율적입니다. one-hot 인코딩을 사용할 경우, 손실 계산이 약간 비효율적일 수 있습니다.
요약
One-hot 인코딩은 Cross-Entropy Loss에서도 사용할 수 있으며, 각 클래스의 손실을 독립적으로 평가할 수 있습니다.
그러나 nn.CrossEntropyLoss를 사용할 때는 raw logits을 입력으로 사용해야 하므로, one-hot 인코딩을 사용할 경우 nn.BCELoss와 같은 손실 함수를 사용하는 것이 좋습니다.
'''

'''
티 클래스 객체의 경계를 명확하게 검출하기 위해 사용할 수 있는 여러 가지 방법이 있습니다. 이 방법들은 모델의 성능을 향상시키고, 객체의 경계를 보다 정확하게 정의하는 데 도움을 줄 수 있습니다. 아래는 몇 가지 주요 방법입니다:

1. 경계 감지 알고리즘
Canny Edge Detection: 경계 감지 알고리즘으로, 이미지에서 경계를 찾는 데 효과적입니다. 세그멘테이션 후 후처리 단계로 사용될 수 있습니다.
Sobel Filter: 이미지의 기울기를 계산하여 경계를 강조하는 방법입니다.
2. Post-processing Techniques
Conditional Random Fields (CRF): 세그멘테이션 결과를 개선하기 위해 CRF를 사용하여 픽셀 간의 관계를 모델링하고 경계를 부드럽게 조정합니다.
Morphological Operations: 침식, 팽창 등의 형태학적 연산을 사용하여 경계를 명확하게 하고 노이즈를 제거합니다.
3. Loss Function Adjustments
Boundary-aware Loss Functions: 경계 감지에 초점을 맞춘 손실 함수를 사용하여 모델이 경계를 더 잘 학습하도록 유도합니다. 예를 들어, Dice Loss 또는 Focal Loss와 같은 손실 함수를 사용할 수 있습니다.
4. Multi-task Learning
Segmentation + Boundary Detection: 객체 세그멘테이션과 경계 감지를 동시에 학습하는 멀티태스크 학습을 통해 모델이 더 나은 경계 인식을 할 수 있도록 합니다. 예를 들어, 경계 마스크를 추가로 학습시켜 경계를 더 잘 인식하도록 할 수 있습니다.
5. Data Augmentation
Augmenting Boundaries: 경계가 뚜렷한 데이터를 증강하여 모델이 다양한 경계 상황을 학습하도록 합니다. 예를 들어, 이미지의 경계에 노이즈를 추가하거나 경계를 강조하는 변형을 적용할 수 있습니다.
6. Attention Mechanisms
Attention-based Models: 경계에 집중할 수 있도록 설계된 어텐션 메커니즘을 사용하는 모델을 구현합니다. 이는 모델이 중요한 경계 정보를 더 잘 학습하도록 도와줍니다.
7. Ensemble Methods
Model Ensembling: 여러 모델의 예측을 결합하여 경계를 더 정확하게 검출합니다. 각 모델이 다른 특성을 학습하게 하여 최종 결과의 정확성을 높일 수 있습니다.
8. Post-Processing with Contours
Contour Detection: 세그멘테이션 결과에서 윤곽선을 추출하여 객체의 경계를 명확하게 표시합니다. OpenCV의 findContours 함수를 사용할 수 있습니다.
'''

'''
import torch
import torch.nn as nn
import torch.optim as optim

# U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        # U-Net 구조 정의 (예: 인코더, 디코더 블록 등)
        # ...

    def forward(self, x):
        # U-Net의 순전파 과정
        # ...
        return x  # 최종 출력

# 가중치 마스크 생성 함수
def create_weight_mask(targets, class_weights):
    """
    Create a weight mask based on the target classes.
    
    Args:
        targets (Tensor): Ground truth masks (N, H, W).
        class_weights (list): List of weights for each class.
    
    Returns:
        Tensor: Weight mask (N, H, W).
    """
    weight_mask = torch.zeros_like(targets, dtype=torch.float32)
    for cls in range(len(class_weights)):
        weight_mask[targets == cls] = class_weights[cls]  # 특정 클래스에 대해 가중치 설정
    return weight_mask

# 손실 함수 정의 (가중치 적용)
def weighted_cross_entropy_loss(outputs, targets, weight_mask):
    """
    Calculate the weighted cross-entropy loss.
    
    Args:
        outputs (Tensor): Model outputs (N, C, H, W).
        targets (Tensor): Ground truth masks (N, H, W).
        weight_mask (Tensor): Weight mask (N, H, W).
    
    Returns:
        Tensor: Calculated loss.
    """
    # CrossEntropyLoss는 logits를 입력으로 받기 때문에 softmax를 적용할 필요 없음
    criterion = nn.CrossEntropyLoss(reduction='none')  # 'none'으로 설정하여 각 픽셀의 손실을 개별적으로 계산
    loss = criterion(outputs, targets)  # 각 픽셀에 대한 손실 계산
    loss = loss * weight_mask  # 가중치 마스크 적용
    return loss.mean()  # 평균 손실 반환

# 데이터셋 초기화
# dataset = SegmentationDataset(image_paths, mask_paths, num_classes=4)

# 클래스 가중치 설정 (예시)
class_weights = [1.0, 2.0, 3.0, 4.0]  # 각 클래스에 대한 가중치

# 모델, 옵티마이저 초기화
model = UNet(in_channels=3, out_channels=4)  # 입력 채널과 클래스 수 설정
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
for epoch in range(num_epochs):
    for images, targets in dataloader:
        outputs = model(images)  # 모델의 출력
        weight_mask = create_weight_mask(targets, class_weights)  # 가중치 마스크 생성

        # 손실 계산
        loss = weighted_cross_entropy_loss(outputs, targets, weight_mask)  # 가중치가 적용된 손실 계산

        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 역전파
        optimizer.step()  # 가중치 업데이트

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

'''