import cv2
import numpy as np

image = cv2.imread('image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (480, 480))

_, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

contours, _ = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

h, w = image.shape[:2]
dst = np.zeros((h, w, 3), np.uint8)

for i in range(len(contours)):
    #BGR
    bgr_color = (194, 117, 0)
    cv2.drawContours(dst, contours, i, bgr_color, 1, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.imshow('image_bin', image_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()