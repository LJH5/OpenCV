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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# 이미지 로드 및 외곽선 검출
img = cv2.VideoCapture(0) # 웹캠 입력

while True:
    ret, frame = img.read()

    if not ret:
        break

    contours = detect_contour(frame)

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
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # 실제 중심점
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)  # 예측 중심점
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2) # 외곽선 표시

    cv2.imshow("Kalman Filter Contour Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

img.release()
cv2.destroyAllWindows()