import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('contour/image/cat.png')
image = cv2.resize(image, (500, 500))

image_copy = image.copy()
image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

_, image_bin = cv2.threshold(image_copy, 0, 255, cv2.THRESH_OTSU)

# 윤곽선 검출
image_bin = cv2.bitwise_not(image_bin)
contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # 원래 윤곽선 그리기 (파란색)
    cv2.drawContours(image, [contour], -1, (255, 0, 0), 2)

    # 윤곽선 단순화 (epsilon 값 조절)
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
