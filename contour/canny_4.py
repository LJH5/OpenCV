import cv2, sys
import numpy as np

image_path = "contour/image/cat.png"  # 적절한 경로로 수정 필요

# 이미지 불러오기 (그레이스케일 변환)
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: 이미지 파일을 찾을 수 없습니다.")
    sys.exit()

# cv2.THRESH_OTSU 임계값으로 low, high 설정
ret, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
low = 0.5 * ret  # Otsu 값의 절반을 Low로1
high = 1.5 * ret  # Otsu 값의 1.5배를 High로
edges_1 = cv2.Canny(image, low, high)

# 이미지 픽셀의 중간값으로 low, high 설정
median_value = np.median(image)
low = int(max(0, 0.5 * median_value))
high = int(min(255, 1.5 * median_value))
edges_2 = cv2.Canny(image, low, high)

cv2.imshow('Edges_1', edges_1)
cv2.imshow('Edges_2', edges_2)

cv2.waitKey(0)
cv2.destroyAllWindows()