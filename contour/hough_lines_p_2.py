import cv2
import numpy as np
import random

def find_parallel_lines(image_path, angle_tolerance=5):
    # 이미지 로드 및 전처리
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not open or find the image at {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, image_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    image_bin = cv2.bitwise_not(image_bin)

    # 윤곽선 검출 및 그리기
    contours, _ = cv2.findContours(image_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour_image = np.zeros_like(gray)
    cv2.drawContours(contour_image, contours, -1, (194, 117, 0), 1)

    # 직선 검출
    lines = cv2.HoughLinesP(contour_image, 1, (np.pi / 180), 20, minLineLength=30, maxLineGap=10)

    if lines is not None and len(lines) > 1:
        # 직선의 기울기 각도 저장
        slopes = []
        parallel_groups = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                slope = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
            else:
                slope = 90 if y2 > y1 else -90  # 수직선 처리
            slopes.append(slope)

        for i, slope1 in enumerate(slopes):
            group_found = False
            for group in parallel_groups:
                if abs(slope1 - slopes[group[0]]) < angle_tolerance:
                    group.append(i)
                    group_found = True
                    break
            if not group_found:
                parallel_groups.append([i])

        # 평행선 그룹별로 색상 지정 및 그리기
        for i, group in enumerate(parallel_groups):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 랜덤 색상 생성
            for line_index in group:
                x1, y1, x2, y2 = lines[line_index][0]
                cv2.line(image, (x1, y1), (x2, y2), color, 2)

        # 결과 표시
        cv2.imshow("Parallel Lines", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No lines or not enough lines detected.")

# 함수 실행
find_parallel_lines("contour/image/bolt.png")