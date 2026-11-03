from pathlib import Path
import argparse
import csv

import cv2
import numpy as np

from common import CentroidTracker


BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "data" / "vtest.mp4"
OUTPUT_DIR = BASE_DIR / "output"

WIDTH = 640
HEIGHT = 360

MIN_AREA = 500
HISTORY = 150
VAR_THRESHOLD = 25
MAX_DISTANCE = 60
MAX_MISSED = 10
WARMUP_FRAMES = 30


def main(no_gui=False):
    OUTPUT_DIR.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(
            f"영상 열기 실패: {VIDEO_PATH}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    model = cv2.createBackgroundSubtractorMOG2(
        history=HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=True
    )

    tracker = CentroidTracker(
        max_distance=MAX_DISTANCE,
        max_missed=MAX_MISSED
    )

    k3 = np.ones((3, 3), np.uint8)
    k7 = np.ones((7, 7), np.uint8)

    video_output = OUTPUT_DIR / "tracking_result.avi"
    csv_output = OUTPUT_DIR / "tracking_results.csv"

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(video_output),
        fourcc,
        fps,
        (WIDTH, HEIGHT)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("결과 영상 저장 실패")

    frame_number = 0

    print("원본 FPS:", fps)
    print("영상 크기:", WIDTH, "x", HEIGHT)
    print("추적 시작 (종료: q)")

    try:
        with open(
            csv_output,
            "w",
            newline="",
            encoding="utf-8"
        ) as csv_file:

            csv_writer = csv.writer(csv_file)

            csv_writer.writerow([
                "frame", "time_s", "track_id",
                "cx", "cy", "x", "y", "w", "h"
            ])

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                frame = cv2.resize(
                    frame, (WIDTH, HEIGHT)
                )

                # 1. MOG2 전경 마스크 생성
                raw = model.apply(frame)

                # 2. 그림자 후보 제거
                _, mask = cv2.threshold(
                    raw, 200, 255,
                    cv2.THRESH_BINARY
                )

                # 3. 형태학적 연산
                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_OPEN, k3
                )

                mask = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, k7
                )

                boxes = []

                # 4. 초기 30프레임 배경 적응
                if frame_number >= WARMUP_FRAMES:

                    contours, _ = cv2.findContours(
                        mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        area = cv2.contourArea(contour)

                        if area < MIN_AREA:
                            continue

                        x, y, w, h = cv2.boundingRect(
                            contour
                        )

                        boxes.append((x, y, w, h))

                # 5. 중심점 기반 ID 추적
                visible = tracker.update(boxes)

                view = frame.copy()

                # 6. 추적 결과 시각화 및 CSV 기록
                for track_id, track in visible.items():
                    x, y, w, h = track["box"]
                    cx, cy = track["center"]

                    cv2.rectangle(
                        view,
                        (x, y),
                        (x + w, y + h),
                        (0, 255, 0),
                        2
                    )

                    cv2.circle(
                        view,
                        (cx, cy),
                        4,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        view,
                        f"ID {track_id}",
                        (x, max(20, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    # 이동 궤적 표시
                    trail = track["trail"]

                    for i in range(1, len(trail)):
                        cv2.line(
                            view,
                            trail[i - 1],
                            trail[i],
                            (255, 255, 0),
                            2
                        )

                    csv_writer.writerow([
                        frame_number,
                        round(frame_number / fps, 3),
                        track_id,
                        cx, cy,
                        x, y, w, h
                    ])

                cv2.putText(
                    view,
                    f"Frame: {frame_number}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    view,
                    f"Visible: {len(visible)}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # 7. 결과 영상 저장
                writer.write(view)

                # 8. 화면 출력
                if not no_gui:
                    cv2.imshow("Tracking", view)
                    cv2.imshow("Foreground Mask", mask)

                    delay = max(1, round(1000 / fps))

                    if (
                        cv2.waitKey(delay) & 0xFF
                        == ord("q")
                    ):
                        break

                frame_number += 1

    finally:
        cap.release()
        writer.release()

        if not no_gui:
            cv2.destroyAllWindows()

    print("추적 완료!")
    print("영상 저장:", video_output)
    print("CSV 저장:", csv_output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="화면 표시 없이 파일 저장"
    )

    args = parser.parse_args()

    main(no_gui=args.no_gui)
