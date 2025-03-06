import cv2
import numpy as np

# 칼만 필터 초기화
kalman = cv2.KalmanFilter(4, 2)  # 상태 변수: 4개 (x, y, dx, dy), 측정값: 2개 (x, y)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1
kalman.errorCovPost = np.eye(4, dtype=np.float32) * 1

# 외곽선 검출 함수 (예시)
def detect_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # 그래디언트
    grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
    cv2.imshow("Gradient Magnitude", magnitude)

    # canny
    image_canny = cv2.Canny(magnitude, 15, 50)
    cv2.imshow("image_canny", image_canny)

    # 엣지 연결을 위한 팽창 연산 적용
    kernel = np.ones((5,5), np.uint8)
    edges_dilated = cv2.dilate(image_canny, kernel, iterations=2)
    cv2.imshow("Edge Dilated", edges_dilated)

    # 닫힘 연산 적용
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Edge Closed", edges_closed)

    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 이미지 로드 및 외곽선 검출
image = cv2.imread("contour/image/relay.png")

image = cv2.resize(image, (300, 300))

while True:
    image_copy = image.copy()
    contours = detect_contour(image_copy)

    if contours:
        contour = max(contours, key=cv2.contourArea) # 가장 큰 외곽선 선택
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # 칼만 필터 예측 및 보정
            measurement = np.array([[np.float32(cx)], [np.float32(cy)]])
            kalman.correct(measurement)
            prediction = kalman.predict()
            px, py = int(prediction[0]), int(prediction[1])

            # 결과 표시
            cv2.circle(image_copy, (cx, cy), 5, (0, 0, 255), -1)  # 실제 중심점
            cv2.circle(image_copy, (px, py), 5, (0, 255, 0), -1)  # 예측 중심점
            cv2.drawContours(image_copy, [contour], -1, (255, 0, 0), 2) # 외곽선 표시

    cv2.imshow("Kalman Filter Contour Tracking", image_copy)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cv2.destroyAllWindows()