import cv2
import numpy as np

# 이미지 로드
image = cv2.imread("contour/image/geometry.png")

# 직선 검출
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
cv2.imshow("edges", edges)
lines = cv2.HoughLinesP(edges, 5, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

# 결과 표시
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()