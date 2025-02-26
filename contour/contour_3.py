import random, cv2
import numpy as np

image = cv2.imread('contour/image/cat.png', cv2.IMREAD_GRAYSCALE)
_, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
contours, _ = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

h, w = image.shape[:2]
dst = np.zeros((h, w, 3), np.uint8)

for i in range(len(contours)):
    points = contours[i]  # 외곽선을 그릴 객체의 포인트 행렬
    area = cv2.contourArea(points)  # 객체의 넓이 계산
    if area > 600:
        #외곽선 그리기
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(dst, contours, i, c, 1, cv2.LINE_AA)

        #외곽선으로 모멘트 계산
        m = cv2.moments(points)

        #외곽선의 중심점 좌표
        x = m['m10']/m['m00']
        y = m['m01']/m['m00']
        cv2.circle(dst, (int(x),int(y)), 3, c, -1)

        #외곽선 둘레 * 0.01
        p1 = 0.01 * cv2.arcLength(points, True)
        #외곽선 근사화(점의 수를 줄임)
        ap = cv2.approxPolyDP(points, p1, True)
        c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        #계산된 근사치 좌표로 외곽선 그림
        cv2.drawContours(dst, [ap], 0, c, 1, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.imshow('image_bin', image_bin)
cv2.imshow('dst', dst)
cv2.waitKey()
cv2.destroyAllWindows()