import cv2, sys
import numpy as np

def canny_contour(image):

    if image is None:
        print("Error: 이미지 파일을 찾을 수 없습니다.")
        sys.exit()

    cv2.imshow("Original Image", image)

    # 흑백
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러
    image_blur = cv2.GaussianBlur(image_gray, (7, 7), 0)
    cv2.imshow("image_blur", image_blur)

    # 그래디언트
    grad_x = cv2.Sobel(image_blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image_blur, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))  # 정규화
    cv2.imshow("Gradient Magnitude", magnitude)

    # canny
    image_canny = cv2.Canny(magnitude, 5, 70)
    cv2.imshow("image_canny", image_canny)

    # 엣지 연결을 위한 팽창 연산 적용
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(image_canny, kernel, iterations=1)
    cv2.imshow("Edge Dilated", edges_dilated)

    # 닫힘 연산 적용
    edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("Edge Closed", edges_closed)

    # 윤곽선 검출
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours
