
from pathlib import Path
import cv2

video = Path(__file__).resolve().parent / "data" / "vtest.mp4"

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise RuntimeError("영상 열기 실패")

fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print("FPS:", fps)
print("영상 크기:", int(width), "x", int(height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Video", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()