from pathlib import Path
import cv2

video = Path(__file__).resolve().parent / "data" / "vtest.mp4"
cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError("영상 열기 실패")

previous = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # 1. 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 가우시안 블러
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. 이전 프레임과 차이 계산
    if previous is not None:
        delta = cv2.absdiff(gray, previous)

        # 4. 임계값으로 이진화 (임계값 15, 25, 40 비교해보기)
        _, mask = cv2.threshold(
            delta, 25, 255, cv2.THRESH_BINARY
        )

        cv2.imshow("Original", frame)
        cv2.imshow("Frame Difference", mask)

    # 5. 현재 프레임을 이전 프레임으로 저장
    previous = gray.copy()

    # 약 30FPS에 맞춰 재생
    if cv2.waitKey(33) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
