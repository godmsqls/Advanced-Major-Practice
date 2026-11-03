
from pathlib import Path
import cv2
import numpy as np

video = Path(__file__).resolve().parent / "data" / "vtest.mp4"
cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError("영상 열기 실패")

# 1. MOG2 배경 모델 생성
model = cv2.createBackgroundSubtractorMOG2(
    history=150,
    varThreshold=25,
    detectShadows=True
)

# 2. 형태학적 연산 커널
k3 = np.ones((3, 3), np.uint8)
k5 = np.ones((7, 7), np.uint8)

frame_count = 0

# 최소 윤곽선 면적 -> 200, 500, 1200 조절
min_area = 200

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # 원본 비율 16:9 유지
    frame = cv2.resize(frame, (640, 360))
    frame_count += 1

    # 3. MOG2 배경 차분
    raw = model.apply(frame)

    # 4. 그림자 후보 제거
    _, mask = cv2.threshold(
        raw, 200, 255, cv2.THRESH_BINARY
    )

    # 5. 열림 연산: 작은 잡음 제거
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, k3
    )

    # 6. 닫힘 연산: 작은 구멍 메우기
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, k5
    )

    # 초기 30프레임은 배경 학습
    if frame_count <= 30:
        cv2.imshow("Original", frame)
        cv2.imshow("MOG2 Mask", mask)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break
        continue

    # 7. 윤곽선 검출
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 8. 최소 면적 이상의 영역에 상자 표시
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            frame, (x, y), (x + w, y + h),
            (0, 255, 0), 2
        )

    cv2.imshow("Original", frame)
    cv2.imshow("MOG2 Mask", mask)

    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
