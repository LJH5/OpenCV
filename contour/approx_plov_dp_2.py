import cv2
import numpy as np
from canny_4 import Canny_Edge

# 이미지 불러오기
image = cv2.imread('contour/image/objects2.png')

image_copy = image.copy()

blured_image = cv2.GaussianBlur(image_copy,(3, 3), 3)
cv2.imshow("blured image", blured_image)

edge_1, edge_2 = Canny_Edge(blured_image, 0.2, 0.6)
cv2.imshow("edge 1", edge_1)
cv2.imshow("edge 2", edge_2)

# 엣지 연결을 위한 팽창 연산 적용
kernel = np.ones((3,3), np.uint8)
edges_dilated = cv2.dilate(edge_2, kernel, iterations=1)
cv2.imshow("Edge Dilated", edges_dilated)

# 닫힘 연산 적용
edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Edge Closed", edges_closed)

contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(edge_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    # contour을 리스트로 넣어 윤곽선 한번에 다 그리기
    cv2.drawContours(image_copy, [contour], -1, (255, 0, 0), 1)

    # 윤곽선 단순화, epsilon 값 작을수록 촘촘
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 단순화된 윤곽선 그리기 (초록색)
    cv2.drawContours(image_copy, [approx], -1, (0, 255, 0), 1)

    # 꼭짓점 좌표 출력
    for point in approx:
        x, y = point[0]
        cv2.circle(image_copy, (x, y), 5, (0, 0, 255), -1)

# 결과 출력
cv2.imshow("Origin", image)
cv2.imshow("Approximated Contours", image_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()
