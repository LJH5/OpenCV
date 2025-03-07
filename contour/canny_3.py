import cv2, sys
import numpy as np

def canny_contour(image, low_threshold, high_threshold):

    if image is None:
        print("Error: 이미지 파일을 찾을 수 없습니다.")
        sys.exit()

    cv2.imshow("Original Image", image)

    # 1. 이미지 그레이스케일 변환
    image_copy = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)

    # 2. 노이즈 제거 (가우시안 블러)
    blurred = cv2.GaussianBlur(image_copy, (5, 5), 1.4)
    cv2.imshow("Gaussian Blur", blurred)

    # 3. 그래디언트 계산 (Sobel 필터 적용)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude_blurred = np.sqrt(grad_x**2 + grad_y**2)
    magnitude_blurred = np.uint8(255 * magnitude_blurred / np.max(magnitude_blurred))  # 정규화
    cv2.imshow("Gradient Magnitude_blurred", magnitude_blurred)

    # 4. canny로 edge 검출
    edges = cv2.Canny(magnitude_blurred, low_threshold, high_threshold)

    # 5. 외곽선 검출
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours
