import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('contour/image/relay.png')
image = cv2.resize(image, (500, 500))

image_copy = image.copy()

# 흑백
image_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

# 블러
image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)


# 그래디언트
grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(grad_x**2 + grad_y**2)
magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
cv2.imshow("Gradient Magnitude", magnitude)

# canny
image_canny = cv2.Canny(magnitude, 0, 50)
cv2.imshow("image_canny", image_canny)


# 윤곽선 검출
contours, _ = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # contour을 리스트로 넣어 윤곽선 한번에 다 그리기
    cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)

    # 윤곽선 단순화, epsilon 값 작을수록 촘촘
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 단순화된 윤곽선 그리기 (초록색)
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)

    # 꼭짓점 좌표 출력
    for point in approx:
        x, y = point[0]
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)

# 결과 출력
cv2.imshow("Approximated Contours", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
