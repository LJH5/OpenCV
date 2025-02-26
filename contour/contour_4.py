import random, cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
_, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

contours, hierachy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
	hull = cv2.convexHull(cnt) #convex hull 추출
	cont_img = cv2.drawContours(image, [hull], 0, (0,0,255), 2)

cv2.imshow('image', image)
cv2.imshow('image_bin', image_bin)
cv2.waitKey()
cv2.destroyAllWindows()