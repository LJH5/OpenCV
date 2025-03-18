import cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

# cv2.threshold(src, thresh, maxval, type, dst=None)

# cv2.THRESH_OTSU: thresh 자동으로 처리, thresh 이상 maxval로 변경
ret_1, image_bin_1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

# cv2.THRESH_BINARY: thresh보다 큰 픽셀은 maxval, 작은 픽셀은 0
ret_2, image_bin_2 = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

# cv2.THRESH_MASK: 흑색 이미지로 변경
ret_3, image_bin_3 = cv2.threshold(image, 0, 255, cv2.THRESH_MASK)

# cv2.THRESH_TOZERO: thresh보다 작은 픽셀은 0
ret_4, image_bin_4 = cv2.threshold(image, 80, 255, cv2.THRESH_TOZERO)

# cv2.THRESH_TRIANGLE: triangle 알고리즘, thresh 자동으로 처리, thresh 이상 maxval로 변경
ret_5, image_bin_5 = cv2.threshold(image, 0, 255, cv2.THRESH_TRIANGLE)

# cv2.THRESH_TRUNC: thresh보다 높은 픽셀을 thresh로 변경, 이하는 원본값
ret_6, image_bin_6 = cv2.threshold(image, 200, 255, cv2.THRESH_TRUNC)


cv2.imshow('Orign', image)
cv2.imshow('THRESH_OTSU', image_bin_1)
cv2.imshow('THRESH_BINARY', image_bin_2)
cv2.imshow('THRESH_MASK', image_bin_3)
cv2.imshow('THRESH_TOZERO', image_bin_4)
cv2.imshow('THRESH_TRIANGLE', image_bin_5)
cv2.imshow('THRESH_TRUNC', image_bin_6)

cv2.waitKey()
cv2.destroyAllWindows()