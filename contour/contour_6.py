import cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', image)

# 타겟 구역 복사하기
x1, x2, y1, y2 = 70, 90, 55, 75
target_area = image[y1:y2, x1:x2].copy()
cv2.imshow('target_area', target_area)

# 타겟 구역의 크기 저장
target_h, target_w = target_area.shape[:2]

# 타겟 구역 확대
target_area = cv2.resize(target_area, (300, 300))

# 이미지 이진화
_, image_bin = cv2.threshold(target_area, 0, 255, cv2.THRESH_OTSU)
cv2.imshow('image_bin', image_bin)

# canny
image_canny = cv2.Canny(image_bin, 0, 255)
cv2.imshow("image_canny", image_canny)

# down sizing
image_canny = cv2.resize(image_canny, (target_h, target_w))
cv2.imshow("image_canny_downsizing", image_canny)

# 외곽선 그리기
contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for i in range(len(contours)):
    #BGR
    bgr_color = (194, 117, 0)

    cv2.drawContours(target_area, contours, i, bgr_color, 2, cv2.LINE_AA)

cv2.imshow('target_area_draw', target_area)

cv2.waitKey()
cv2.destroyAllWindows()