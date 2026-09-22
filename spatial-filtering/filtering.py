import cv2
import numpy as np

# imread() 이미지 불러오기
image = cv2.imread('./images/input.jpg')

# -----[평균 필터]-----
# 커널 크기는 홀수만 사용!!!
# blur3 = cv2.blur(image, (3, 3))
# blur5 = cv2.blur(image, (5, 5))
# blur9 = cv2.blur(image, (9, 9))

# cv2.imshow('3x3', blur3)
# cv2.imshow('5x5', blur5)
# cv2.imshow('9x9', blur9)

# -----[가우시안 필터]-----
# average = cv2.blur(image, (5, 5))

# # (5, 5) 커널 크기, 0 표준편차
# gaussian = cv2.GaussianBlur(image, (5, 5), 0)

# cv2.imshow('Average', average)
# cv2.imshow('Gaussian', gaussian)

# -----[미디언 필터]-----
# gaussian = cv2.GaussianBlur(image, (5, 5), 0)
# median = cv2.medianBlur(image, 5)

# cv2.imshow('Gaussian', gaussian)
# cv2.imshow('Median', median)

# -----[샤프닝 필터]-----
kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])

# filter2D() 커널을 영상에 적용
# -1: 입력 영상과 같은 데이터 형식 유지
sharpened = cv2.filter2D(image, -1, kernel)
cv2.imshow('Sharpened',sharpened)

# -----[출력]-----

# imshow() 원본 영상 화면에 출력
# cv2.imshow('Original', image)

# waitKey() 키 입력까지 대기, 이거 없으면 창 떴다가 바로 사라짐
cv2.waitKey(0)

# destroyAllWindows() 창 닫기
cv2.destroyAllWindows()