import cv2

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

_, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
threshold = cv2.bitwise_not(threshold)

adaptive_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
adaptive_threshold = cv2.bitwise_not(adaptive_threshold)

contours_1, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours_2, _ = cv2.findContours(adaptive_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('image', image)
cv2.imshow('threshold', threshold)
cv2.imshow('adaptive_threshold', adaptive_threshold)

cv2.waitKey()
cv2.destroyAllWindows()