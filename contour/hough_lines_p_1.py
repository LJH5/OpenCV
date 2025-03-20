import cv2
import numpy as np
import random

# 이미지 로드
image = cv2.imread("contour/image/bolt.png")
image_copy = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, image_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

image_bin = cv2.bitwise_not(image_bin)

contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

contour_image = np.zeros_like(gray)

# 외곽선 그리기
cv2.drawContours(contour_image, contours, -1, (194, 117, 0), 1)

# 직선 검출
lines = cv2.HoughLinesP(contour_image, 1, (np.pi / 180), 2, minLineLength=30, maxLineGap=20)

# 결과 표시
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 랜덤 색상 생성
        cv2.line(image_copy, (x1, y1), (x2, y2), color, 2)

cv2.imshow("Origin", image)
cv2.imshow("Detected Contours", contour_image)
cv2.imshow("Detected Lines", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()