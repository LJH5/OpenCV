import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('contour/image/relay.png')

# 이미지 확대하기
h, w = image.shape[:2]
image_copy = cv2.resize(image, (300, 300))

# 흑백
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

# 블러
image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)
cv2.imshow("image_blur", image_blur)

# 그래디언트
grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
cv2.imshow("Gradient Magnitude", magnitude)

# canny
image_canny = cv2.Canny(magnitude, 5, 70)
cv2.imshow("image_canny", image_canny)

# 엣지 연결을 위한 팽창 연산 적용
kernel = np.ones((5,5), np.uint8)
edges_dilated = cv2.dilate(image_canny, kernel, iterations=3)
cv2.imshow("Edge Dilated", edges_dilated)

# 닫힘 연산 적용
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Edge Closed", edges_closed)

# 윤곽선 검출
contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # contour을 리스트로 넣어 윤곽선 한번에 다 그리기
    cv2.drawContours(image_copy, [contour], -1, (255, 0, 0), 2)

    # 윤곽선 단순화, epsilon 값 작을수록 촘촘
    approx = cv2.approxPolyDP(contour, 80, True)

    # 단순화된 윤곽선 그리기 (초록색)
    cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 2)

    # 꼭짓점 좌표 출력
    for point in approx:
        x, y = point[0]
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)

# 결과 출력
cv2.imshow("Approximated Contours", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
