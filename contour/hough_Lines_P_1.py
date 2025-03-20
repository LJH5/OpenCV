import cv2
import numpy as np
from canny_3 import canny_contour

# 이미지 로드
image = cv2.imread("contour/image/relay.png")

# 외곽선 검출
contours = canny_contour(image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contour_image = np.zeros_like(gray)

# 외곽선 그리기
cv2.drawContours(contour_image, contours, -1, (194, 117, 0), 1)
cv2.imshow("Detected Contours", contour_image)

# 직선 검출
lines = cv2.HoughLinesP(contour_image, 1, (np.pi / 180), 9, minLineLength=20, maxLineGap=50)

# 결과 표시
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Detected Lines", image)
cv2.waitKey(0)
cv2.destroyAllWindows()