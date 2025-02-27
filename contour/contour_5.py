import cv2
import numpy as np

# 이미지 불러오기 (그레이스케일)
image = cv2.imread("contour/image/objects.png", cv2.IMREAD_GRAYSCALE)

# 2. 노이즈 제거 (가우시안 블러)
blurred = cv2.GaussianBlur(image, (5,5), 0)
cv2.imshow("Gaussian Blur", blurred)

# 3. 그래디언트 계산 (Sobel 필터 적용)
grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
cv2.imshow("Gradient Magnitude", magnitude)

# Canny 엣지 검출
edges = cv2.Canny(magnitude, 100, 150)
cv2.imshow("Canny Edge", edges)

# ⬇ 엣지 연결을 위한 팽창 연산 적용
kernel = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edges, kernel, iterations=1)
cv2.imshow("Edge Dilated", edges_dilated)

# ⬇ 닫힘 연산 (Closing) 적용
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Edge Closed", edges_closed)

# 허프 변환을 이용한 선 검출
lines = cv2.HoughLinesP(edges_closed, 1, np.pi/180, threshold=50, minLineLength=10, maxLineGap=10)
cv2.imshow("Lines", edges_closed)

# ⬇ 외곽선 찾기 및 그리기
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_contours = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
cv2.drawContours(image_contours, contours, -1, (194, 117, 0), 1)

for cnt in contours:
	hull = cv2.convexHull(cnt) #convex hull 추출
	cont_img = cv2.drawContours(image, [hull], 0, (194, 117, 0), 1)

# 결과 출력
cv2.imshow("Contours", image_contours)
cv2.imshow('cont_img', cont_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
