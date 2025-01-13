import torch
import torch.onnx
import onnx
import onnxruntime
import numpy as np

def convert_to_onnx(model, sample_input, save_path='model.onnx'):
    """
    PyTorch 모델을 ONNX 형식으로 변환합니다.
    
    Args:
        model: PyTorch 모델
        sample_input: 모델 입력 예시 (torch.Tensor)
        save_path: 저장할 ONNX 파일 경로
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 모델 변환
    torch.onnx.export(
        model,                    # 변환할 PyTorch 모델
        sample_input,            # 모델의 입력값 예시
        save_path,               # 저장할 경로
        export_params=True,      # 모델 파라미터 저장
        opset_version=11,        # ONNX 버전
        do_constant_folding=True,# 상수 폴딩 최적화 사용
        input_names=['input'],   # 입력 이름
        output_names=['output'], # 출력 이름
        dynamic_axes={           # 동적 축 설정 (배치 크기를 가변적으로)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"모델이 {save_path}로 성공적으로 변환되었습니다.")

def verify_onnx_model(onnx_path):
    """
    변환된 ONNX 모델을 검증합니다.
    
    Args:
        onnx_path: ONNX 모델 파일 경로
    """
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX 모델 검증이 완료되었습니다.")
        return True
    except Exception as e:
        print(f"ONNX 모델 검증 중 오류 발생: {str(e)}")
        return False

class ONNXInference:
    """ONNX 모델을 사용하여 추론을 수행하는 클래스"""
    
    def __init__(self, onnx_path):
        """
        Args:
            onnx_path: ONNX 모델 파일 경로
        """
        self.session = onnxruntime.InferenceSession(onnx_path)
    
    def inference(self, input_data):
        """
        ONNX 모델을 사용하여 추론을 수행합니다.
        
        Args:
            input_data: 입력 데이터 (numpy array 또는 torch.Tensor)
            
        Returns:
            numpy array: 모델의 출력값
        """
        # torch.Tensor를 numpy로 변환
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
        
        # 입력 이름 가져오기
        input_name = self.session.get_inputs()[0].name
        
        # 추론 실행
        ort_inputs = {input_name: input_data}
        ort_outputs = self.session.run(None, ort_inputs)
        
        return ort_outputs[0]

# 사용 예시
"""
# 1. 모델 변환
model = YourModel()  # 여러분의 PyTorch 모델
sample_input = torch.randn(1, input_channels, height, width)  # 모델 입력 형태에 맞는 샘플
convert_to_onnx(model, sample_input, 'your_model.onnx')

# 2. 모델 검증
verify_onnx_model('your_model.onnx')

# 3. 추론
onnx_inference = ONNXInference('your_model.onnx')
input_data = torch.randn(1, input_channels, height, width)  # 실제 입력 데이터
output = onnx_inference.inference(input_data)
"""

'''
import onnxruntime
import numpy as np

# ONNX 모델 로드
onnx_model_path = 'your_model.onnx'
ort_session = onnxruntime.InferenceSession(onnx_model_path)

# 입력 데이터 준비 (예: 랜덤 데이터 또는 실제 테스트 데이터)
input_data = np.random.randn(1, input_channels, height, width).astype(np.float32)  # 모델의 입력 형태에 맞게 조정

# 추론 수행
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: input_data}
ort_outputs = ort_session.run(None, ort_inputs)

# 결과 확인
print(ort_outputs[0])  # 모델의 예측 결과
'''