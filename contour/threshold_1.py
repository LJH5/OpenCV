import cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

_, image_bin_1 = cv2.threshold(image, 175, 255, cv2.THRESH_OTSU)
_, image_bin_2 = cv2.threshold(image, 175, 255, cv2.THRESH_BINARY)
_, image_bin_3 = cv2.threshold(image, 175, 255, cv2.THRESH_MASK)
_, image_bin_4 = cv2.threshold(image, 175, 255, cv2.THRESH_TOZERO)
_, image_bin_5 = cv2.threshold(image, 175, 255, cv2.THRESH_TRIANGLE)
_, image_bin_6 = cv2.threshold(image, 175, 255, cv2.THRESH_TRUNC)


cv2.imshow('Orign', image)
cv2.imshow('THRESH_OTSU', image_bin_1)
cv2.imshow('THRESH_BINARY', image_bin_2)
cv2.imshow('THRESH_MASK', image_bin_3)
cv2.imshow('THRESH_TOZERO', image_bin_4)
cv2.imshow('THRESH_TRIANGLE', image_bin_5)
cv2.imshow('THRESH_TRUNC', image_bin_6)

cv2.waitKey()
cv2.destroyAllWindows()