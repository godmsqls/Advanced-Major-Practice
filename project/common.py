import math

class CentroidTracker:
    def __init__(self, max_distance=60, max_missed=10):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}

    def get_center(self, box):
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        return (cx, cy)

    def get_distance(self, p1, p2):
        return math.hypot(
            p2[0] - p1[0],
            p2[1] - p1[1]
        )

    def register(self, box):
        center = self.get_center(box)
        track_id = self.next_id

        self.tracks[track_id] = {
            "id": track_id,
            "center": center,
            "box": box,
            "missed": 0,
            "trail": [center]
        }

        self.next_id += 1

    def update(self, boxes):
        boxes = [tuple(map(int, b)) for b in boxes]
        centers = [self.get_center(b) for b in boxes]

        # 기존 추적 객체가 없으면 모두 신규 등록
        if not self.tracks:
            for box in boxes:
                self.register(box)
            return self.get_visible()

        # 기존 객체와 새 검출 영역의 거리 후보
        candidates = []

        for track_id, track in self.tracks.items():
            for i, center in enumerate(centers):
                distance = self.get_distance(
                    track["center"], center
                )

                if distance <= self.max_distance:
                    candidates.append(
                        (distance, track_id, i)
                    )

        # 거리가 짧은 후보부터 연결
        candidates.sort(key=lambda x: x[0])

        matched_tracks = set()
        matched_boxes = set()

        for distance, track_id, box_index in candidates:
            if track_id in matched_tracks:
                continue

            if box_index in matched_boxes:
                continue

            track = self.tracks[track_id]
            center = centers[box_index]

            track["center"] = center
            track["box"] = boxes[box_index]
            track["missed"] = 0
            track["trail"].append(center)

            # 궤적은 최근 30개 중심점만 유지
            if len(track["trail"]) > 30:
                track["trail"].pop(0)

            matched_tracks.add(track_id)
            matched_boxes.add(box_index)

        # 연결되지 않은 기존 객체 처리
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["missed"] += 1

                if (
                    self.tracks[track_id]["missed"]
                    > self.max_missed
                ):
                    del self.tracks[track_id]

        # 연결되지 않은 검출 영역은 새 ID 등록
        for i, box in enumerate(boxes):
            if i not in matched_boxes:
                self.register(box)

        return self.get_visible()

    def get_visible(self):
        return {
            track_id: track
            for track_id, track in self.tracks.items()
            if track["missed"] == 0
        }
