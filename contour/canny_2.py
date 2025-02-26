import cv2, sys
import numpy as np

image_path = "contour/image/cat.png"  # 적절한 경로로 수정 필요

# 1. 이미지 불러오기 (그레이스케일 변환)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: 이미지 파일을 찾을 수 없습니다.")
    sys.exit()

cv2.imshow("Original Image", image)

# # 2. 노이즈 제거 (가우시안 블러)
# blurred = cv2.GaussianBlur(image, (5, 5), 1.4)
# cv2.imshow("Gaussian Blur", blurred)

# 3. 그래디언트 계산 (Sobel 필터 적용)
grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
cv2.imshow("Gradient Magnitude", magnitude)

# 4. 비최대 억제 (Canny 내부적으로 처리됨, 따로 구현 가능)
# 5. 이중 임계값 적용 및 히스테리시스 엣지 연결
edges = cv2.Canny(image, 50, 150)  # 임계값 설정 (조정 가능)
cv2.imshow("Canny Edge Detection", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()