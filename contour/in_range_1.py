import cv2
import numpy as np

# 1. 이미지 로드
image = cv2.imread("contour/image/tomato.png")

# 2. BGR에서 HSV로 변환
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 3. 특정 색상 범위 지정 (예: 빨간색)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])

# 4. 범위 내 픽셀 찾기
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask = cv2.bitwise_or(mask1, mask2)  # 두 마스크를 합침

# 5. 특정 색상만 검은색(0), 나머지는 흰색(255)으로 변환
result = cv2.bitwise_not(mask)  # 마스크 반전

# 6. 결과 출력
cv2.imshow("Origin", image)
cv2.imshow("Hsv", hsv)
cv2.imshow("Mask1", mask1)
cv2.imshow("Mask2", mask2)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
