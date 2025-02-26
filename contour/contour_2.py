import cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

_, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

image_bin = cv2.bitwise_not(image_bin)

contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

h, w = image.shape[:2]
# 이미지 크기의 검은 배경 만들기, 3채널, 픽셀값 8비트 정수로 저장
dst = np.zeros((h, w, 3), np.uint8)
# 흰 배경
wst = np.ones((h, w, 3), np.uint8) * 255

for i in range(len(contours)):
    #BGR
    bgr_color = (194, 117, 0)
    # 검은 배경에 외곽선 그리기
    cv2.drawContours(dst, contours, i, bgr_color, 2, cv2.LINE_AA)
    # 흰 배경에 외곽선 그리기
    cv2.drawContours(wst, contours, i, bgr_color, 2, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.imshow('image_bin', image_bin)
cv2.imshow('dst', dst)
cv2.imshow('wst', wst)
cv2.waitKey()
cv2.destroyAllWindows()