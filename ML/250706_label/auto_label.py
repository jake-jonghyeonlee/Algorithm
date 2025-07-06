import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from skimage import filters, segmentation, measure
from scipy import ndimage
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

class StandardUNet(nn.Module):
    """표준 U-Net 구조 (더 깊고 정확한 모델)"""
    def __init__(self, in_channels=3, out_channels=1, base_channels=64):
        super(StandardUNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, base_channels)
        self.enc2 = self.conv_block(base_channels, base_channels*2)
        self.enc3 = self.conv_block(base_channels*2, base_channels*4)
        self.enc4 = self.conv_block(base_channels*4, base_channels*8)
        
        # Bottleneck
        self.bottleneck = self.conv_block(base_channels*8, base_channels*16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = self.conv_block(base_channels*16, base_channels*8)
        
        self.upconv3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = self.conv_block(base_channels*8, base_channels*4)
        
        self.upconv2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = self.conv_block(base_channels*4, base_channels*2)
        
        self.upconv1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = self.conv_block(base_channels*2, base_channels)
        
        # Final layer
        self.final = nn.Conv2d(base_channels, out_channels, 1)
        
    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return torch.sigmoid(self.final(dec1))

class SemiAutoDataset(Dataset):
    """Semi-auto labeling을 위한 데이터셋"""
    def __init__(self, images, masks, transforms=None, boundary_weight=None):
        self.images = images
        self.masks = masks
        self.transforms = transforms
        self.boundary_weight = boundary_weight
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert to tensor if not already
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float().unsqueeze(0)
            
        sample = {'image': image, 'mask': mask}
        
        if self.boundary_weight is not None:
            sample['boundary_weight'] = torch.from_numpy(self.boundary_weight[idx]).float()
            
        return sample

class AdvancedSemiAutoLabelingTool:
    """향상된 Semi-auto labeling 도구"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = StandardUNet(base_channels=64).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        # Data augmentation
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.3),
            A.ElasticTransform(p=0.3),
        ])
        
        # 정확도 추적을 위한 변수들
        self.accuracy_history = []
        self.best_accuracy = 0.0
        self.best_masks = None
        
    def extract_boundary_canny(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Canny edge detection을 사용한 boundary 추출"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Image gradient 계산
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Mask boundary 추출 (Canny)
        mask_binary = (mask > 0.5).astype(np.uint8)
        mask_boundary = cv2.Canny(mask_binary * 255, 30, 100)
        
        # Gradient와 mask boundary 결합
        combined_boundary = cv2.bitwise_and(
            (gradient_magnitude > np.percentile(gradient_magnitude, 60)).astype(np.uint8) * 255,
            cv2.dilate(mask_boundary, np.ones((3,3), np.uint8), iterations=1)
        )
        
        return combined_boundary / 255.0
    
    def extract_boundary_morphology(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Morphology subtraction을 사용한 boundary 추출"""
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        eroded = cv2.erode(mask_binary, kernel, iterations=1)
        boundary = mask_binary - eroded
        
        # 이미지 gradient와 결합
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Gradient 정보로 boundary 정제
        gradient_mask = (gradient_magnitude > np.percentile(gradient_magnitude, 50)).astype(np.uint8)
        refined_boundary = cv2.bitwise_and(boundary * 255, gradient_mask * 255)
        
        return refined_boundary / 255.0
    
    def extract_boundary_hybrid(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Canny와 Morphology를 결합한 hybrid 방법"""
        canny_boundary = self.extract_boundary_canny(image, mask)
        morph_boundary = self.extract_boundary_morphology(image, mask)
        
        # 두 방법을 결합 (OR 연산)
        combined = np.logical_or(canny_boundary > 0.1, morph_boundary > 0.1).astype(np.float32)
        
        return combined
    
    def generate_pseudo_labels(self, images: List[np.ndarray], 
                             rough_masks: List[np.ndarray]) -> List[np.ndarray]:
        """향상된 pseudo label 생성"""
        pseudo_labels = []
        
        for img, mask in zip(images, rough_masks):
            # GrabCut을 사용한 정교한 세그멘테이션
            if len(img.shape) == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # 초기 마스크 설정
            mask_gc = np.zeros(img_bgr.shape[:2], np.uint8)
            mask_gc[(mask > 0.7)] = cv2.GC_FGD  # 확실한 전경
            mask_gc[(mask < 0.3)] = cv2.GC_BGD  # 확실한 배경
            mask_gc[((mask >= 0.3) & (mask <= 0.7))] = cv2.GC_PR_FGD  # 추정 전경
            
            # GrabCut 적용
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            
            try:
                cv2.grabCut(img_bgr, mask_gc, None, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_MASK)
                pseudo_label = np.where((mask_gc == 2) | (mask_gc == 0), 0, 1).astype(np.float32)
            except:
                # GrabCut 실패시 Watershed 사용
                if len(img.shape) == 3:
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img
                    
                dist_transform = cv2.distanceTransform((mask > 0.5).astype(np.uint8), 
                                                     cv2.DIST_L2, 5)
                sure_fg = (dist_transform > 0.5 * dist_transform.max()).astype(np.uint8)
                sure_bg = cv2.dilate(mask.astype(np.uint8), np.ones((3,3), np.uint8), iterations=3)
                
                markers = measure.label(sure_fg)
                markers = segmentation.watershed(-dist_transform, markers, mask=sure_bg)
                pseudo_label = (markers > 0).astype(np.float32)
            
            pseudo_labels.append(pseudo_label)
            
        return pseudo_labels
    
    def calculate_advanced_boundary_weights(self, masks: List[np.ndarray], 
                                          images: List[np.ndarray]) -> List[np.ndarray]:
        """향상된 boundary 가중치 계산"""
        boundary_weights = []
        
        for mask, img in zip(masks, images):
            # 3가지 boundary 방법 모두 사용
            canny_boundary = self.extract_boundary_canny(img, mask)
            morph_boundary = self.extract_boundary_morphology(img, mask)
            hybrid_boundary = self.extract_boundary_hybrid(img, mask)
            
            # 가중치 맵 초기화
            weight_map = np.ones_like(mask)
            
            # 각 boundary 방법에 따른 가중치 부여
            weight_map[canny_boundary > 0.1] = 4.0    # Canny boundary: 4배
            weight_map[morph_boundary > 0.1] = 3.0    # Morphology boundary: 3배
            weight_map[hybrid_boundary > 0.1] = 5.0   # Hybrid boundary: 5배
            
            # Distance transform으로 boundary 주변 영역 가중치
            boundary_combined = np.logical_or(
                np.logical_or(canny_boundary > 0.1, morph_boundary > 0.1),
                hybrid_boundary > 0.1
            ).astype(np.uint8)
            
            dist_from_boundary = ndimage.distance_transform_edt(1 - boundary_combined)
            
            # 더 정교한 거리 가중치 (sigmoid 함수 사용)
            distance_weight = 1 / (1 + np.exp((dist_from_boundary - 3) / 2))
            weight_map = weight_map * (1 + 2 * distance_weight)
            
            # 불확실한 영역 (0.3 < mask < 0.7)에 추가 가중치
            uncertain_mask = ((mask > 0.3) & (mask < 0.7)).astype(np.float32)
            weight_map = weight_map * (1 + 2 * uncertain_mask)
            
            boundary_weights.append(weight_map)
            
        return boundary_weights
    
    def calculate_accuracy(self, pred_masks: List[np.ndarray], 
                          true_masks: List[np.ndarray]) -> float:
        """정확도 계산 (IoU 사용)"""
        total_iou = 0.0
        
        for pred, true in zip(pred_masks, true_masks):
            pred_binary = (pred > 0.5).astype(np.uint8)
            true_binary = (true > 0.5).astype(np.uint8)
            
            intersection = np.logical_and(pred_binary, true_binary).sum()
            union = np.logical_or(pred_binary, true_binary).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0 if intersection == 0 else 0.0
                
            total_iou += iou
            
        return total_iou / len(pred_masks)
    
    def weighted_loss(self, pred, target, weight=None):
        """향상된 가중치 손실 함수"""
        # Binary Cross Entropy Loss
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        # Dice Loss 추가
        smooth = 1e-7
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        # 가중치 적용
        if weight is not None:
            bce_loss = bce_loss * weight
            
        # 두 손실 결합
        combined_loss = bce_loss.mean() + 0.3 * dice_loss
        
        return combined_loss
    
    def train_stage(self, images: List[np.ndarray], 
                   masks: List[np.ndarray], 
                   epochs: int, 
                   stage_name: str,
                   boundary_weights: List[np.ndarray] = None) -> Tuple[List[np.ndarray], float]:
        """단계별 학습 함수"""
        print(f"{stage_name} 학습 시작...")
        
        dataset = SemiAutoDataset(images, masks, transforms=self.transforms, 
                                boundary_weight=boundary_weights)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        self.model.train()
        best_epoch_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                images_batch = batch['image'].to(self.device)
                masks_batch = batch['mask'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images_batch)
                
                # 가중치 적용
                if boundary_weights is not None:
                    weights_batch = batch['boundary_weight'].to(self.device)
                    loss = self.weighted_loss(outputs, masks_batch, weights_batch)
                else:
                    loss = self.weighted_loss(outputs, masks_batch)
                
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            self.scheduler.step(avg_loss)
            
            if avg_loss < best_epoch_loss:
                best_epoch_loss = avg_loss
                
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # 예측 결과 생성
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for img in images:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
                img_tensor = img_tensor.to(self.device)
                pred = self.model(img_tensor)
                pred_np = pred.squeeze().cpu().numpy()
                predictions.append(pred_np)
        
        return predictions, best_epoch_loss
    
    def run_iterative_semi_auto_labeling(self, images: List[np.ndarray], 
                                        initial_masks: List[np.ndarray],
                                        max_iterations: int = 5,
                                        early_stopping_patience: int = 3) -> List[np.ndarray]:
        """반복적 semi-auto labeling 프로세스"""
        print("=== 반복적 Semi-Auto Labeling 프로세스 시작 ===")
        
        current_masks = initial_masks.copy()
        best_masks = current_masks.copy()
        best_accuracy = 0.0
        patience_counter = 0
        
        for iteration in range(max_iterations):
            print(f"\n{'='*20} Iteration {iteration + 1} {'='*20}")
            
            # 1단계: Rough mask 학습
            rough_predictions, rough_loss = self.train_stage(
                images, current_masks, epochs=40, stage_name="1단계: Rough mask"
            )
            
            # 2단계: Boundary 추출 (3가지 방법 모두 사용)
            print("2단계: 다중 방법 boundary 추출...")
            
            # 3단계: Pseudo label 생성 및 boundary 중심 학습
            print("3단계: 향상된 pseudo label 생성...")
            pseudo_labels = self.generate_pseudo_labels(images, rough_predictions)
            
            boundary_predictions, boundary_loss = self.train_stage(
                images, pseudo_labels, epochs=35, stage_name="3단계: Boundary 중심"
            )
            
            # 4단계: 고급 boundary 가중치 계산 및 재학습
            print("4단계: 고급 boundary 가중치 계산...")
            boundary_weights = self.calculate_advanced_boundary_weights(boundary_predictions, images)
            
            final_predictions, final_loss = self.train_stage(
                images, boundary_predictions, epochs=45, stage_name="4단계: 가중치 재학습",
                boundary_weights=boundary_weights
            )
            
            # 정확도 계산 (이전 iteration과 비교)
            if iteration > 0:
                accuracy = self.calculate_accuracy(final_predictions, current_masks)
                print(f"현재 정확도 (IoU): {accuracy:.4f}")
                
                # 최고 정확도 갱신
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_masks = final_predictions.copy()
                    patience_counter = 0
                    print(f"새로운 최고 정확도! {best_accuracy:.4f}")
                else:
                    patience_counter += 1
                    print(f"정확도 개선 없음. Patience: {patience_counter}/{early_stopping_patience}")
                
                self.accuracy_history.append(accuracy)
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping 조건 만족. 최고 정확도: {best_accuracy:.4f}")
                    break
            
            current_masks = final_predictions
            
            # 진행 상황 출력
            print(f"Iteration {iteration + 1} 완료")
            print(f"  - Rough Loss: {rough_loss:.4f}")
            print(f"  - Boundary Loss: {boundary_loss:.4f}")
            print(f"  - Final Loss: {final_loss:.4f}")
        
        print(f"\n=== 반복적 Semi-Auto Labeling 완료 ===")
        print(f"최종 최고 정확도: {best_accuracy:.4f}")
        print(f"총 {len(self.accuracy_history)} 회 반복 수행")
        
        return best_masks if best_masks is not None else current_masks
    
    def visualize_comprehensive_results(self, images: List[np.ndarray], 
                                      initial_masks: List[np.ndarray],
                                      final_masks: List[np.ndarray]):
        """포괄적인 결과 시각화"""
        n_images = len(images)
        fig, axes = plt.subplots(n_images, 5, figsize=(25, 5*n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i, (img, init_mask, final_mask) in enumerate(zip(images, initial_masks, final_masks)):
            # 원본 이미지
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # 초기 mask
            axes[i, 1].imshow(img)
            axes[i, 1].imshow(init_mask, alpha=0.5, cmap='Reds')
            axes[i, 1].set_title('Initial Mask')
            axes[i, 1].axis('off')
            
            # 최종 mask
            axes[i, 2].imshow(img)
            axes[i, 2].imshow(final_mask, alpha=0.5, cmap='Blues')
            axes[i, 2].set_title('Final Mask')
            axes[i, 2].axis('off')
            
            # Boundary 비교
            canny_boundary = self.extract_boundary_canny(img, final_mask)
            morph_boundary = self.extract_boundary_morphology(img, final_mask)
            
            axes[i, 3].imshow(canny_boundary, cmap='hot')
            axes[i, 3].set_title('Canny Boundary')
            axes[i, 3].axis('off')
            
            axes[i, 4].imshow(morph_boundary, cmap='hot')
            axes[i, 4].set_title('Morphology Boundary')
            axes[i, 4].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 정확도 히스토리 플롯
        if self.accuracy_history:
            plt.figure(figsize=(10, 6))
            plt.plot(self.accuracy_history, 'b-o', linewidth=2, markersize=8)
            plt.title('Accuracy History (IoU)', fontsize=16)
            plt.xlabel('Iteration', fontsize=14)
            plt.ylabel('IoU Score', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.show()

# 사용 예제
def example_usage():
    """향상된 사용 예제"""
    
    def create_sample_data():
        # 더 현실적인 샘플 데이터 생성
        np.random.seed(42)
        
        # 복잡한 형태의 샘플 이미지
        img1 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # 더 복잡한 초기 mask
        mask1 = np.zeros((256, 256), dtype=np.float32)
        # 원형 객체
        center = (128, 128)
        radius = 50
        y, x = np.ogrid[:256, :256]
        mask1[(x - center[0])**2 + (y - center[1])**2 <= radius**2] = 1.0
        
        mask2 = np.zeros((256, 256), dtype=np.float32)
        # 사각형 객체
        mask2[80:180, 80:180] = 1.0
        # 일부 노이즈 추가
        mask2[mask2 > 0] += np.random.normal(0, 0.1, mask2[mask2 > 0].shape)
        mask2 = np.clip(mask2, 0, 1)
        
        return [img1, img2], [mask1, mask2]
    
    # 데이터 준비
    images, initial_masks = create_sample_data()
    
    # 향상된 Semi-auto labeling 도구 초기화
    tool = AdvancedSemiAutoLabelingTool()
    
    # 반복적 semi-auto labeling 실행
    final_masks = tool.run_iterative_semi_auto_labeling(
        images, initial_masks, 
        max_iterations=7,  # 최대 7번 반복
        early_stopping_patience=3  # 3번 연속 개선 없으면 중단
    )
    
    # 포괄적인 결과 시각화
    tool.visualize_comprehensive_results(images, initial_masks, final_masks)
    
    return final_masks

if __name__ == "__main__":
    print("향상된 Semi-Auto Labeling Tool 실행 예제")
    print("- 표준 U-Net 모델 사용")
    print("- Canny vs Morphology boundary 비교")
    print("- 반복적 정확도 향상 프로세스")
    print("-" * 50)
    
    final_masks = example_usage()
    
    print(f"\n최종 결과:")
    print(f"- 처리된 mask 개수: {len(final_masks)}")
    print(f"- 각 mask 크기: {[mask.shape for mask in final_masks]}")
    print(f"- 각 mask 범위: {[(mask.min(), mask.max()) for mask in final_masks]}")
