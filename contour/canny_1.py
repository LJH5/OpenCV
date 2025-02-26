import cv2
import numpy as np

# 이미지 불러오기 (그레이스케일)
image = cv2.imread("contour/image/cat.png", cv2.IMREAD_GRAYSCALE)

# Canny 엣지 검출 적용 (임계값: low, high)
edges1 = cv2.Canny(image, 0, 255)
edges2 = cv2.Canny(image, 100, 200)
edges3 = cv2.Canny(image, 150, 200)
edges4 = cv2.Canny(image, 150, 250)

# 결과 출력
cv2.imshow("Original Image", image)
cv2.imshow("Canny Edge Detection1", edges1)
cv2.imshow("Canny Edge Detection2", edges2)
cv2.imshow("Canny Edge Detection3", edges3)
cv2.imshow("Canny Edge Detection4", edges4)

cv2.waitKey(0)
cv2.destroyAllWindows()
