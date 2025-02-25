import cv2, sys, random

# grayscale로 이미지 불러오기
image = cv2.imread('contour/image/geometry.png', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (500, 500))

# 이미지 불러오기 실해하면 종료
if image is None:
    print("image read fail")
    sys.exit()

# 외곽선 탐지 모드들
modes = [cv2.RETR_TREE, cv2.RETR_LIST, cv2.RETR_EXTERNAL, cv2.RETR_CCOMP]
name = ['RETR_TREE', 'RETR_LIST', 'RETR_EXTERNAL', 'RETR_CCOMP']

for i in range (len(modes)):
    # 외곽선 탐지
    contours, hierarchy = cv2.findContours(image, modes[i], cv2.CHAIN_APPROX_NONE)
    # 회색조 -> BRG
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # 외곽선 계층
    contour_idx = 0
    while contour_idx >= 0:
        # 랜덤색
        contour_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        # 외곽선 그리기
        cv2.drawContours(image_rgb, contours, contour_idx, contour_color, 5, cv2.LINE_8, hierarchy)
        # contour_idx의 다음 외곽선 번호 반환, 없으면 -1 반환
        contour_idx = hierarchy[0, contour_idx, 0]
    cv2.imshow(name[i], image_rgb)

# 키 입력 대기 ms, 0은 무한
cv2.waitKey(0)
# imshow 종료
cv2.destroyAllWindows()