import torch
import torch.nn as nn
import numpy as np

class SimpleDiffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        """
        간단한 Diffusion 모델 구현
        Args:
            n_steps: diffusion steps의 수
            beta_start: 시작 노이즈 스케줄def extract_noise(self, x_0, noisy_image, t):

                alpha_t = self.alpha_bar[t]
                
                # forward process 공식을 역으로 계산
                # noisy_image = √(αt)x_0 + √(1-αt)noise
                # 따라서, noise = (noisy_image - √(αt)x_0) / √(1-αt)
                extracted_noise = (noisy_image - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
                return extracted_noise
            beta_end: 끝 노이즈 스케줄
        """
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def forward_process(self, x_0, t):
        """
        Forward process: 점진적으로 노이즈를 추가
        Args:
            x_0: 원본 이미지
            t: 타임스텝
        """
        alpha_t = self.alpha_bar[t]
        noise = torch.randn_like(x_0)
        
        # forward process 공식: √(αt)x_0 + √(1-αt)ε
        noisy_image = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
        return noisy_image, noise
    
    def reverse_process(self, x_t, t, model):
        """
        Reverse process: 노이즈를 제거하고 원본 이미지를 복원
        Args:
            x_t: 노이즈가 있는 이미지
            t: 타임스텝
            model: 노이즈를 예측하는 UNet 모델
        """
        predicted_noise = model(x_t, t)
        alpha_t = self.alpha_bar[t]
        
        # reverse process 공식
        x_0 = (x_t - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
        return x_0

    def extract_noise(self, x_0, noisy_image, t):
        """
        이미지에서 노이즈를 추출하는 메소드
        Args:
            x_0: 원본 이미지
            noisy_image: 노이즈가 추가된 이미지
            t: 타임스텝
        Returns:
            추출된 노이즈
        """
        alpha_t = self.alpha_bar[t]
        
        # forward process 공식을 역으로 계산
        # noisy_image = √(αt)x_0 + √(1-αt)noise
        # 따라서, noise = (noisy_image - √(αt)x_0) / √(1-αt)
        extracted_noise = (noisy_image - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
        return extracted_noise

    def apply_style_noise(self, target_image, style_noise, t):
        """
        타겟 이미지에 스타일 노이즈를 적용하는 메소드
        Args:
            target_image: 스타일을 적용할 대상 이미지
            style_noise: 다른 이미지에서 추출한 노이즈
            t: 타임스텝
        Returns:
            스타일이 적용된 이미지
        """
        alpha_t = self.alpha_bar[t]
        
        # forward process 공식을 사용하여 스타일 노이즈 적용
        styled_image = torch.sqrt(alpha_t) * target_image + torch.sqrt(1 - alpha_t) * style_noise
        return styled_image

if __name__ == "__main__":
    # Diffusion 모델 초기화
    diffusion = SimpleDiffusion()
    
    # 1. 이미지 준비
    # 스타일을 가져올 원본 이미지 (예: 반 고흐의 별이 빛나는 밤)
    style_image = torch.randn(1, 3, 32, 32)  # 실제로는 이미지를 로드해야 함
    
    # 스타일을 적용할 타겟 이미지 (예: 풍경 사진)
    target_image = torch.randn(1, 3, 32, 32)  # 실제로는 이미지를 로드해야 함
    
    # 2. 스타일 이미지에서 노이즈 추출
    # 여러 타임스텝에서 노이즈를 추출하면 더 다양한 스타일 특성을 캡처할 수 있음
    t_steps = [200, 500, 800]  # 다양한 타임스텝에서 시도
    style_noises = []
    
    for t in map(torch.tensor, t_steps):
        # 스타일 이미지에 노이즈 추가
        noisy_style, _ = diffusion.forward_process(style_image, t)
        
        # 노이즈 추출
        style_noise = diffusion.extract_noise(style_image, noisy_style, t)
        style_noises.append(style_noise)
    
    # 3. 추출한 노이즈를 타겟 이미지에 적용
    styled_images = []
    for t, noise in zip(map(torch.tensor, t_steps), style_noises):
        # 각 타임스텝에서 스타일 노이즈 적용
        styled_image = diffusion.apply_style_noise(target_image, noise, t)
        styled_images.append(styled_image)
    
    # 4. 결과 확인 (실제로는 이미지를 시각화해야 함
    print("스타일 전이 결과:")
    for t, img in zip(t_steps, styled_images):
        print(f"타임스텝 {t}에서의 스타일 전이 이미지 shape: {img.shape}")