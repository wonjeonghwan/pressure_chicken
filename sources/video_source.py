"""
카메라 / 영상 파일 입력 추상화

사용:
    src = VideoSource({"type": "file", "path": "video_a.mp4"})
    src = VideoSource({"type": "camera", "index": 0})

    with src:
        ret, frame = src.read()
"""

import sys
import time
import threading
import cv2
import numpy as np


class VideoSource:
    """카메라 또는 파일을 동일한 인터페이스로 제공"""

    _RECONNECT_THRESHOLD = 10   # 연속 실패 N회 시 재연결 시도
    _RECONNECT_COOLDOWN  = 5.0  # 재연결 시도 최소 간격 (초) — MSMF 해제 안정화 여유
    _STALE_THRESHOLD     = 5.0  # 마지막 프레임으로부터 이 시간(초) 초과 시 stale 처리
    _DEFAULT_FRAME_WIDTH  = 1920  # 카메라 요청 해상도 (미지원 시 아래 폴백 해상도로 순차 시도)
    _DEFAULT_FRAME_HEIGHT = 1080
    _RESOLUTION_FALLBACKS = ((1280, 720), (640, 480))  # 요청 해상도로 프레임을 못 읽을 때 순서대로 재시도

    def __init__(self, source_cfg: dict):
        self._cfg = source_cfg
        self._cap: cv2.VideoCapture | None = None
        self.failed = False  # 열기 실패 여부
        self._fail_count   = 0
        self._last_reconnect = 0.0

        # 캡처 스레드 (카메라 소스 전용)
        self._capture_thread: threading.Thread | None = None
        self._stop_event: threading.Event = threading.Event()
        self._frame_lock: threading.Lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0

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
            self._cap = self._open_camera_with_fallback(index)

        if not self._cap.isOpened():
            print(f"[VideoSource] 경고: 소스 열기 실패 ({self._cfg}). 빈 프레임으로 동작합니다.")
            self._cap = None
            self.failed = True
            return

        # 카메라 노출 설정 (카메라 소스 + exposure 키 있을 때만)
        if src_type == "camera" and "exposure" in self._cfg:
            # 자동 노출 끄기 (값은 카메라마다 다름: 보통 1=수동, 3=자동)
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            result = self._cap.set(cv2.CAP_PROP_EXPOSURE, self._cfg["exposure"])
            actual = self._cap.get(cv2.CAP_PROP_EXPOSURE)
            if result:
                print(f"[VideoSource] 노출 설정 완료: {actual}")
            else:
                print(f"[VideoSource] 노출 설정 미지원 (카메라 드라이버 불가) — gamma로 대체 권장")

    def _open_camera_cv(self, index: int, width: int, height: int) -> cv2.VideoCapture:
        """지정 해상도로 cv2.VideoCapture 생성 (isOpened 여부는 호출부에서 확인)."""
        if sys.platform == "win32":
            # CAP_PROP_HW_ACCELERATION=NONE: MSMF D3D11 HW 컬러 변환 파이프라인 비활성화
            # — Nvidia 환경에서 일부 USB 카메라 드라이버와 조합 시 YUV→BGR 밝기 과도 상승(블리칭) 버그 방지
            return cv2.VideoCapture(
                index, cv2.CAP_MSMF,
                [
                    cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE,
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                ],
            )
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap

    def _open_camera_with_fallback(self, index: int) -> cv2.VideoCapture:
        """요청 해상도(기본 1920x1080)로 열기 시도 → 실제 프레임을 못 읽으면 더 작은 표준 해상도로 순차 폴백.

        일부 카메라/드라이버 조합은 미지원 해상도 요청 시 isOpened()는 True를 반환하면서도
        read()가 계속 실패하는 경우가 있어, get()으로 협상 결과만 확인하는 것으로는 불충분함.
        """
        width = self._cfg.get("frame_width", self._DEFAULT_FRAME_WIDTH)
        height = self._cfg.get("frame_height", self._DEFAULT_FRAME_HEIGHT)
        primary = (width, height)
        fallbacks = [(w, h) for (w, h) in self._RESOLUTION_FALLBACKS if w < width]
        candidates = [primary] + fallbacks

        for requested_w, requested_h in candidates:
            cap = self._open_camera_cv(index, requested_w, requested_h)
            if not cap.isOpened():
                cap.release()
                continue
            ret, _ = cap.read()
            if not ret:
                cap.release()
                continue
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fallback_note = "" if (requested_w, requested_h) == primary else \
                f" (요청 {primary[0]}x{primary[1]} 실패 → {requested_w}x{requested_h}로 폴백)"
            print(f"[VideoSource] 카메라 index={index} 해상도 {actual_w}x{actual_h} 로 열림{fallback_note}")
            return cap

        print(f"[VideoSource] 카메라 index={index}: 시도한 모든 해상도({[primary] + fallbacks})에서 프레임을 읽지 못함")
        return cv2.VideoCapture()  # 닫힌 capture 반환 — 호출부에서 isOpened()==False로 처리

    def _try_reconnect(self) -> None:
        """카메라 재연결 시도 (카메라 소스 전용)."""
        import time
        now = time.monotonic()
        if now - self._last_reconnect < self._RECONNECT_COOLDOWN:
            return
        self._last_reconnect = now
        index = self._cfg.get("index", 0)
        print(f"[VideoSource] 재연결 시도: index={index}")
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            time.sleep(0.5)  # MSMF가 장치 핸들을 완전히 해제할 시간
        cap = self._open_camera_with_fallback(index)
        if cap.isOpened():
            self._cap = cap
            self._fail_count = 0
            self.failed = False
            print(f"[VideoSource] 재연결 성공: index={index}")
        else:
            cap.release()
            self._cap = None
            print(f"[VideoSource] 재연결 실패: index={index}")

    def read(self) -> tuple[bool, np.ndarray | None]:
        """프레임 한 장 읽기. 파일 끝이면 처음부터 루프."""
        if self._cap is None:
            # _cap=None 상태(_try_reconnect 실패 후)에도 재시도 카운터를 계속 증가시켜
            # 쿨다운이 지나면 다시 재연결을 시도한다.
            if self._cfg.get("type", "camera") == "camera":
                self._fail_count += 1
                if self._fail_count >= self._RECONNECT_THRESHOLD:
                    self._try_reconnect()
            return False, None

        ret, frame = self._cap.read()

        if not ret and self._cfg.get("type") == "file":
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()

        # 카메라 소스: 연속 실패 시 재연결
        if self._cfg.get("type", "camera") == "camera":
            if not ret:
                self._fail_count += 1
                if self._fail_count >= self._RECONNECT_THRESHOLD:
                    self._try_reconnect()
            else:
                self._fail_count = 0

        # 파일 소스: target_fps 기준으로 프레임 스킵 (실시간 카메라와 동일한 시간축 유지)
        # _skip_frames = round(video_fps / target_fps), main.py에서 주입
        # grab()은 픽셀 디코딩 없이 프레임 포인터만 전진 → seek보다 훨씬 빠름
        if ret and self._cfg.get("type") == "file":
            skip = self._cfg.get("_skip_frames", 1)
            for _ in range(skip - 1):
                self._cap.grab()

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

            gamma = self._cfg.get("gamma")
            if gamma is not None and gamma != 1.0:
                frame = self._gamma_lut(frame, gamma)

        return ret, frame

    # 감마 보정 LUT (gamma > 1 → 하이라이트 억제 / gamma < 1 → 밝기 증가)
    _lut_cache: dict[float, np.ndarray] = {}

    @classmethod
    def _gamma_lut(cls, frame: np.ndarray, gamma: float) -> np.ndarray:
        if gamma not in cls._lut_cache:
            cls._lut_cache[gamma] = np.array(
                [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
                dtype=np.uint8,
            )
        return cv2.LUT(frame, cls._lut_cache[gamma])

    # ── 캡처 스레드 (카메라 소스 전용) ────────────────────────────────────────

    def start_capture(self, fps: float | None = None) -> None:
        """백그라운드 캡처 스레드 시작. 이미 실행 중이면 무시.

        fps: 캡처 속도 상한 (None이면 카메라 최대속도).
             처리 파이프라인 주기(target_fps)와 맞추면 불필요한 프레임 생성을 줄인다.
        """
        if self._cfg.get("type", "camera") != "camera":
            return
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._capture_fps = fps
        self._stop_event.clear()
        idx = self._cfg.get("index", 0)
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"cam-{idx}",
            daemon=True,
        )
        self._capture_thread.start()

    def stop_capture(self) -> None:
        """캡처 스레드 정지 (release() 전에 호출)."""
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=3.0)
            self._capture_thread = None

    def _capture_loop(self) -> None:
        """백그라운드 캡처 루프 — 최신 프레임을 슬롯에 유지."""
        interval = (1.0 / self._capture_fps) if self._capture_fps else None
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            ret, frame = self.read()
            if ret and frame is not None:
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_ts = time.monotonic()
                if interval:
                    # 처리 주기만큼 대기 — stop 신호도 즉시 수신
                    wait = interval - (time.monotonic() - t0)
                    if wait > 0:
                        self._stop_event.wait(wait)
            else:
                # 실패 시 짧게 대기 — _cap=None 상태에서 CPU 스핀 방지
                self._stop_event.wait(0.033)

    def get_latest(self) -> tuple[bool, np.ndarray | None]:
        """메인루프용 비블로킹 최신 프레임 조회.

        캡처 스레드가 없으면(파일 소스 등) 기존 read()로 폴백.
        """
        if self._capture_thread is None:
            return self.read()
        with self._frame_lock:
            frame = self._latest_frame
            ts    = self._latest_ts
        if frame is None:
            return False, None
        if time.monotonic() - ts > self._STALE_THRESHOLD:
            return False, None
        return True, frame

    # ── 자원 해제 ──────────────────────────────────────────────────────────────

    def release(self) -> None:
        self.stop_capture()  # 스레드 먼저 정지 후 cap 해제
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

    @staticmethod
    def find_available_cameras(max_try: int = 10) -> list[int]:
        """사용 가능한 카메라 인덱스 목록을 반환."""
        import sys
        available = []
        for i in range(max_try):
            if sys.platform == "win32":
                cap = cv2.VideoCapture(
                    i, cv2.CAP_MSMF,
                    [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE],
                )
            else:
                cap = cv2.VideoCapture(i)
            if cap is not None and cap.isOpened():
                available.append(i)
                cap.release()
        return available

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.release()
