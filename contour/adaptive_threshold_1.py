import cv2

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (500, 500))

_, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

adaptive_threshold_gaussian = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

adaptive_threshold_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('image', image)
cv2.imshow('threshold', threshold)
cv2.imshow('adaptive_threshold_gaussian', adaptive_threshold_gaussian)
cv2.imshow('adaptive_threshold_mean', adaptive_threshold_mean)

cv2.waitKey()
cv2.destroyAllWindows()