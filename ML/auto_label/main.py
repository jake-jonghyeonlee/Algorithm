import cv2
import numpy as np
import os

class AutoLabeler:
    def __init__(self):
        self.image = None
        self.edges = None
        self.result = None
        self.window_name = "Auto Labeler"
        self.image_path = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 클릭한 지점에서 flood fill 실행
            mask = np.zeros((self.image.shape[0] + 2, self.image.shape[1] + 2), np.uint8)
            flood_fill_mask = self.edges.copy()
            cv2.floodFill(flood_fill_mask, mask, (x, y), 255)
            
            # flood fill로 채워진 영역 찾기
            filled_area = cv2.subtract(flood_fill_mask, self.edges)
            
            # 결과 이미지에 채워진 영역을 검은색으로 설정
            self.result[filled_area == 255] = 0
            
            # 결과 표시
            cv2.imshow(self.window_name, self.result)
    
    def process_image(self, image_path):
        self.image_path = image_path
        # 이미지 로드 및 전처리
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if self.image is None:
            raise ValueError("이미지를 불러올 수 없습니다.")
        
        # 엣지 검출
        self.edges = cv2.Canny(self.image, 50, 150)
        
        # 결과 이미지 초기화 (흰색 배경)
        self.result = np.ones_like(self.image) * 255
        
        # 엣지를 결과 이미지에 그리기
        self.result[self.edges == 255] = 0
            
        # 윈도우 생성 및 마우스 콜백 설정
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # q를 누르면 종료
                break
            elif key == ord('s'):  # s를 누르면 저장
                self.save_result()
                break
            elif key == ord('r'):  # r를 누르면 리셋
                self.result = np.ones_like(self.image) * 255
                self.result[self.edges == 255] = 0
                cv2.imshow(self.window_name, self.result)
        
        cv2.destroyAllWindows()
        
    def save_result(self):
        # 결과 저장
        input_filename = os.path.basename(self.image_path)
        filename_without_ext = os.path.splitext(input_filename)[0]
        
        # output 폴더 생성
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        # output 폴더에 같은 이름으로 png 저장
        output_path = os.path.join(output_dir, f'{filename_without_ext}.png')
        cv2.imwrite(output_path, self.result)

def main():
    labeler = AutoLabeler()
    labeler.process_image('test.jpg')  # 입력 이미지 경로 예시

if __name__ == "__main__":
    main()
