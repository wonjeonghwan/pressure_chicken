"""
카메라 / 영상 파일 입력 추상화

사용:
    src = VideoSource({"type": "file", "path": "video_a.mp4"})
    src = VideoSource({"type": "camera", "index": 0})

    with src:
        ret, frame = src.read()
"""

import cv2
import numpy as np


class VideoSource:
    """카메라 또는 파일을 동일한 인터페이스로 제공"""

    def __init__(self, source_cfg: dict):
        self._cfg = source_cfg
        self._cap: cv2.VideoCapture | None = None
        self.failed = False  # 열기 실패 여부

    def open(self) -> None:
        src_type = self._cfg.get("type", "camera")

        if src_type == "file":
            path = self._cfg["path"]
            self._cap = cv2.VideoCapture(path)
            if not self._cap.isOpened():
                print(f"[VideoSource] 파일 '{path}' 열기 실패 → 웹캠 index 0 으로 폴백")
                fallback = self._cfg.get("fallback_index", 0)
                self._cap = cv2.VideoCapture(fallback)
        else:
            index = self._cfg.get("index", 0)
            self._cap = cv2.VideoCapture(index)

        if not self._cap.isOpened():
            print(f"[VideoSource] 경고: 소스 열기 실패 ({self._cfg}). 빈 프레임으로 동작합니다.")
            self._cap = None
            self.failed = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """프레임 한 장 읽기. 파일 끝이면 처음부터 루프."""
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()

        if not ret and self._cfg.get("type") == "file":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()

        # 다운스케일: FHD(1920×1080) 초과 시 자동으로 FHD로 축소
        # config에 "resize": [w, h] 지정 시 해당 크기로 강제 조정
        if ret and frame is not None:
            explicit = self._cfg.get("resize")
            if explicit:
                frame = cv2.resize(frame, tuple(explicit), interpolation=cv2.INTER_AREA)
            elif frame.shape[1] > 1920 or frame.shape[0] > 1080:
                scale = min(1920 / frame.shape[1], 1080 / frame.shape[0])
                new_w = int(frame.shape[1] * scale)
                new_h = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return ret, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        if self._cap is None:
            return 30.0
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps > 0 else 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        if self._cap is None:
            return (640, 480)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
