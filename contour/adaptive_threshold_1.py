import cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

_, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
image_bin = cv2.bitwise_not(image_bin)

adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_threshold = cv2.bitwise_not(adaptive_threshold)

contours_1, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_2, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

h, w = image.shape[:2]
# 이미지 크기의 검은 배경 만들기, 3채널, 픽셀값 8비트 정수로 저장
dst_1 = np.zeros((h, w, 3), np.uint8)
dst_2 = np.zeros((h, w, 3), np.uint8)
# 흰 배경
wst_1 = np.ones((h, w, 3), np.uint8) * 255
wst_2 = np.ones((h, w, 3), np.uint8) * 255

for i in range(len(contours_1)):
    #BGR
    bgr_color = (194, 117, 0)
    # 검은 배경에 외곽선 그리기
    cv2.drawContours(dst_1, contours_1, i, bgr_color, 2, cv2.LINE_AA)
    # 흰 배경에 외곽선 그리기
    cv2.drawContours(wst_1, contours_1, i, bgr_color, 2, cv2.LINE_AA)

for i in range(len(contours_2)):
    #BGR
    bgr_color = (194, 117, 0)
    # 검은 배경에 외곽선 그리기
    cv2.drawContours(dst_2, contours_2, i, bgr_color, 2, cv2.LINE_AA)
    # 흰 배경에 외곽선 그리기
    cv2.drawContours(wst_2, contours_2, i, bgr_color, 2, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.imshow('image_bin', image_bin)
cv2.imshow('adaptive_threshold', adaptive_threshold)

cv2.imshow('dst_1', dst_1)
cv2.imshow('wst_1', wst_1)
cv2.imshow('dst_2', dst_2)
cv2.imshow('wst_2', wst_2)

cv2.waitKey()
cv2.destroyAllWindows()