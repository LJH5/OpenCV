import cv2
import numpy as np

# 이미지 로드
image = cv2.imread("contour/image/glass_cup.png")
cv2.imshow("origin", image)

image = cv2.bitwise_not(image)
cv2.imshow("bit_not", image)

# 그레이스케일 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 가우시안 블러 적용
blurred = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("blurred", blurred)

# 그래디언트 계산, Sobel 필터
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
cv2.imshow("Gradient Magnitude", magnitude)

# Canny 엣지 검출
edges = cv2.Canny(blurred, 0, 10)
cv2.imshow("edges", edges)

# 엣지 연결을 위한 팽창 연산 적용
kernel = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=2)
cv2.imshow("Edge Dilated", edges_dilated)

# 닫힘 연산 적용
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Edge Closed", edges_closed)

# 윤곽선 검출
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 원본 이미지에 외곽선 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 결과 출력
cv2.imshow("Detected Contours", image)
cv2.waitKey(0)
cv2.destroyAllWin
