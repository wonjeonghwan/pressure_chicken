"""
주기적 프레임 스냅샷 저장 — 학습 데이터 수집용

캡처 스레드 버퍼(get_latest)에서 복사만 하므로 메인 루프에 영향 없음.
저장 위치: {log_dir}/frames/YYYYMMDD_HHMMSS_cam{id}.jpg
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import cv2


class FrameSaver:
    def __init__(
        self,
        sources: dict,
        log_dir: str = "logs",
        interval_sec: int = 7200,
        jpeg_quality: int = 92,
    ) -> None:
        self._sources       = sources
        self._out_dir       = os.path.join(log_dir, "frames")
        self._interval      = interval_sec
        self._jpeg_quality  = jpeg_quality
        self._stop_event    = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        os.makedirs(self._out_dir, exist_ok=True)
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="frame-saver",
            daemon=True,
        )
        self._thread.start()
        print(f"[FrameSaver] start: interval={self._interval//3600}h, dir={self._out_dir}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def save_now(self) -> list[str]:
        """즉시 모든 카메라 프레임 저장. 저장된 파일 경로 목록 반환."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved = []
        for src_id, vs in self._sources.items():
            ret, frame = vs.get_latest()
            if not ret or frame is None:
                continue
            filename = f"{ts}_cam{src_id}.jpg"
            path = os.path.join(self._out_dir, filename)
            ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
            if ok:
                saved.append(path)
        if saved:
            print(f"[FrameSaver] saved {len(saved)} frames: {ts}")
        else:
            print(f"[FrameSaver] no frames available")
        return saved

    def _loop(self) -> None:
        # 시작 직후 1장 즉시 저장 (시스템 정상 동작 확인용)
        self.save_now()

        while not self._stop_event.wait(self._interval):
            self.save_now()
