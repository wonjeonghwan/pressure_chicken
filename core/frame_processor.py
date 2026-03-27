"""
프레임 처리기

동작 방식:
  - 전체 프레임에 대해 YOLO 1회 추론 (pot_body / pot_weight 바운딩박스 색출)
  - 캘리브레이션 ROI의 중심점과 YOLO pot_body 중심점을 거리 기반 매칭
  - 독점 매칭: 각 weight를 x축 거리 기준 가장 가까운 pot 하나에만 배정
  - NCC(정규화 교차상관) 비교로 진동 판별 (FrameDiffTracker)
    - dark_threshold 불필요 → 전체 grayscale 패턴 비교
    - 평균 제거 → 조명 변화 / 오토 익스포저 무관
    - EMA로 위치/크기 안정화 (카메라 흔들림 보정)
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT
from core.state_machine import BurnerRegistry, BurnerState


# ── 진동 판별기 (NCC + EMA 안정화) ──────────────────────────────────────────

class FrameDiffTracker:
    """
    딸랑이 영역의 NCC(정규화 교차상관) 비교로 진동 판별.

    매 프레임:
      1. EMA 안정화된 위치/크기로 딸랑이 크롭 (카메라 흔들림 보정)
      2. 정사각형 패딩 → 32×32 grayscale resize (INTER_LINEAR)
      3. 표준편차 < MIN_STD 이면 스킵 (너무 균일한 크롭 — 패턴 없음)
      4. 이전 프레임 크롭과 NCC 비교
           NCC = Σ(A-μA)(B-μB) / sqrt(Σ(A-μA)² · Σ(B-μB)²)
           - 평균 제거 → 전역 밝기 변화(오토 익스포저) 무관
           - NCC ≈ 1.0 → 동일 패턴 → 정지
           - NCC 낮아짐 → 패턴 변화 → 진동

    window 프레임 중 trigger 개 이상 NCC < ncc_threshold → 진동 확정.
    """

    _CROP_SIZE  = 32
    _SIZE_ALPHA = 0.15   # 크기 EMA: 느리게 (박스 크기 노이즈 흡수)
    _POS_ALPHA  = 0.5    # 위치 EMA: 빠르게 (카메라 이동 즉시 추적)
    _MIN_STD    = 3.0    # 최소 표준편차 (너무 균일하면 스킵)

    def __init__(self, motion_cfg: dict):
        self._window  = motion_cfg.get("window_frames",  30)
        self._trigger = motion_cfg.get("trigger_frames", 20)
        self._ncc_thr = motion_cfg.get("ncc_threshold",  0.85)

        self._prev_gray: np.ndarray | None = None
        self._cv_hist:   deque[bool]       = deque(maxlen=self._window)
        self._last_ncc:  float             = 1.0

        self._cx: float | None = None
        self._cy: float | None = None
        self._w:  float | None = None
        self._h:  float | None = None
        self._last_centroid: tuple[int, int] | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._cv_hist.clear()
        self._last_ncc  = 1.0
        self._cx = self._cy = self._w = self._h = None
        self._last_centroid = None

    @property
    def last_angle(self) -> float | None:
        """UI 호환용 — NCC 반환 (None이면 아직 비교 없음)"""
        return self._last_ncc if self._prev_gray is not None else None

    @property
    def last_deviation(self) -> float:
        """UI 호환용 — NCC 반환"""
        return self._last_ncc

    @property
    def last_centroid(self) -> tuple[int, int] | None:
        return self._last_centroid

    @property
    def score(self) -> float:
        if not self._cv_hist:
            return 0.0
        return min(1.0, sum(self._cv_hist) / self._trigger)

    @staticmethod
    def _ema(prev: float | None, new: float, alpha: float) -> float:
        return new if prev is None else alpha * new + (1.0 - alpha) * prev

    @staticmethod
    def _get_crop_gray(
        frame: np.ndarray,
        cx: float, cy: float, w: float, h: float,
        crop_size: int,
    ) -> np.ndarray | None:
        """EMA 중심/크기 기준 크롭 → 정사각형 패딩 → crop_size×crop_size grayscale"""
        fh, fw = frame.shape[:2]
        x1c = max(0, int(round(cx - w / 2)))
        y1c = max(0, int(round(cy - h / 2)))
        x2c = min(fw, int(round(cx + w / 2)))
        y2c = min(fh, int(round(cy + h / 2)))
        if x2c <= x1c or y2c <= y1c:
            return None

        gray = cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2GRAY)
        ch, cw = gray.shape
        side   = max(ch, cw)
        pad    = np.zeros((side, side), dtype=np.uint8)
        pad[(side - ch) // 2:(side - ch) // 2 + ch,
            (side - cw) // 2:(side - cw) // 2 + cw] = gray
        return cv2.resize(pad, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        p  = a.astype(np.float32)
        q  = b.astype(np.float32)
        pm = p - p.mean()
        qm = q - q.mean()
        denom = float(np.sqrt((pm ** 2).sum() * (qm ** 2).sum()))
        if denom < 1e-6:
            return 1.0
        return float((pm * qm).sum() / denom)

    def update(
        self,
        frame: np.ndarray,
        _x1: int, _y1: int, _x2: int, _y2: int,
        has_weight: bool = False,
        wx1: int = 0, wy1: int = 0, wx2: int = 0, wy2: int = 0,
    ) -> bool:
        motion = False

        if has_weight and wx2 > wx1 and wy2 > wy1:
            self._cx = self._ema(self._cx, (wx1 + wx2) / 2, self._POS_ALPHA)
            self._cy = self._ema(self._cy, (wy1 + wy2) / 2, self._POS_ALPHA)
            self._w  = self._ema(self._w,  wx2 - wx1,       self._SIZE_ALPHA)
            self._h  = self._ema(self._h,  wy2 - wy1,       self._SIZE_ALPHA)
            self._last_centroid = (int(self._cx), int(self._cy))

            curr_gray = self._get_crop_gray(
                frame, self._cx, self._cy, self._w, self._h, self._CROP_SIZE
            )

            if curr_gray is not None:
                std_val = float(curr_gray.astype(np.float32).std())
                if std_val >= self._MIN_STD and self._prev_gray is not None:
                    prev_std = float(self._prev_gray.astype(np.float32).std())
                    if prev_std >= self._MIN_STD:
                        self._last_ncc = self._ncc(self._prev_gray, curr_gray)
                        if self._last_ncc < self._ncc_thr:
                            motion = True
                self._prev_gray = curr_gray
            # curr_gray=None 이거나 has_weight=False 인 경우 prev_gray 유지
            # (딸랑이가 일시적으로 미감지돼도 이전 크롭 기억 → 재감지 시 즉시 NCC 비교 가능)

        self._cv_hist.append(motion)
        return self._check()

    def _check(self) -> bool:
        hist = list(self._cv_hist)
        if len(hist) < self._window:
            return False
        return sum(hist) >= self._trigger


# ── 프레임 처리기 ───────────────────────────────────────────────────────────

class FrameProcessor:
    """
    VideoSource 들에서 프레임을 읽고 하이브리드 감지 -> 상태머신 갱신.
    """

    def __init__(
        self,
        sources:      dict[int, VideoSource],
        burner_cfgs:  list[dict],
        registry:     BurnerRegistry,
        detector:     BurnerDetector,
        motion_cfg:   dict,
    ):
        self._sources    = sources
        self._registry   = registry
        self._detector   = detector
        self._burner_map = {b["id"]: b for b in burner_cfgs}
        self._match_threshold = 200 # 캘리브레이션 앵커와 실제 밥솥 간 최대 허용 거리(px)

        # 화구별 FrameDiffTracker
        self._trackers: dict[int, FrameDiffTracker] = {
            b["id"]: FrameDiffTracker(motion_cfg)
            for b in burner_cfgs
        }

        # 화구별 캘리브레이션 앵커 포인트 계산
        self._anchors: dict[int, tuple[int, int]] = {}
        for b in burner_cfgs:
            roi = b.get("roi")
            if roi:
                x, y, w, h = roi
                self._anchors[b["id"]] = (x + w // 2, y + h // 2)

        self._frame_cache: dict[int, np.ndarray | None] = {}
        self._prev_states: dict[int, BurnerState] = {
            b["id"]: BurnerState.EMPTY for b in burner_cfgs
        }
        self._body_ttl: dict[int, int] = {b["id"]: 0 for b in burner_cfgs}
        
        # UI 오버레이에 사용하기 위해 마지막으로 매칭된 밥솥/딸랑이 박스 저장
        self.last_matched_boxes:  dict[int, tuple[int, int, int, int]] = {}
        self.last_weight_boxes:   dict[int, tuple[int, int, int, int]] = {}
        self.last_centroids:      dict[int, tuple[int, int]] = {}  # 시각화: 무게중심 좌표
        # 키포인트 (top, bot) — 시각화용
        self.last_keypoints: dict[
            int,
            tuple[tuple[float,float,float], tuple[float,float,float]]
        ] = {}

    def read_frames(self) -> dict[int, np.ndarray]:
        frames: dict[int, np.ndarray | None] = {}
        for src_id, src in self._sources.items():
            ret, frame = src.read()
            frames[src_id] = frame if ret else None
        self._frame_cache = frames
        return {sid: f for sid, f in frames.items() if f is not None}

    def detect_and_update(self) -> None:
        # UI에서 '초기화'를 눌렀다면 bsm.state는 이미 EMPTY. 
        # 이를 이 프레임을 처리하기 전에 먼저 감지하여 완전히 캐시를 비워줍니다 (Race Condition 억제).
        for bid in self._burner_map:
            bsm = self._registry.get(bid)
            if bsm.state == BurnerState.EMPTY and self._prev_states[bid] != BurnerState.EMPTY:
                if bid in self._trackers:
                    self._trackers[bid].reset()
                self._prev_states[bid] = BurnerState.EMPTY
                self.last_matched_boxes.pop(bid, None)
                self.last_weight_boxes.pop(bid, None)
                self.last_keypoints.pop(bid, None)
                self._body_ttl[bid] = 0
                
        frames    = self._frame_cache
        detections: dict[int, tuple[bool, bool]] = {}

        src_burners: dict[int, list[int]] = {}
        for bid, cfg in self._burner_map.items():
            src_burners.setdefault(cfg["source_id"], []).append(bid)

        for src_id, burner_ids in src_burners.items():
            frame = frames.get(src_id)

            if frame is None:
                for bid in burner_ids:
                    detections[bid] = (False, False)
                continue

            # Max-Bounding ROI Crop 계산
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0
            has_roi = False
            
            for bid in burner_ids:
                if bid in self._burner_map and self._burner_map[bid].get("roi"):
                    rx, ry, rw, rh = self._burner_map[bid]["roi"]
                    min_x = min(min_x, rx)
                    min_y = min(min_y, ry)
                    max_x = max(max_x, rx + rw)
                    max_y = max(max_y, ry + rh)
                    has_roi = True
                    
            if has_roi:
                # 여백(Margin) 부여: 화면 잘림 방지
                margin = 50
                fh, fw = frame.shape[:2]
                cx1 = max(0, int(min_x) - margin)
                cy1 = max(0, int(min_y) - margin)
                cx2 = min(fw, int(max_x) + margin)
                cy2 = min(fh, int(max_y) + margin)
                
                crop_frame = frame[cy1:cy2, cx1:cx2]
                dets = self._detector.detect(crop_frame)
                
                # Crop된 좌표계를 원본 좌표계로 복원
                for d in dets:
                    d.x1 += cx1; d.x2 += cx1
                    d.y1 += cy1; d.y2 += cy1
                    if d.keypoints:
                        d.keypoints = [
                            (kx + cx1, ky + cy1, kv)
                            for kx, ky, kv in d.keypoints
                        ]
            else:
                dets = self._detector.detect(frame)
            
            bodies = [d for d in dets if d.class_id == CLASS_POT_BODY and d.confidence >= 0.3]
            weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]
            
            matched_bodies: dict[int, tuple[int, int, int, int]] = {}
            # (has_weight, w_box, kp_top, kp_bot)
            matched_has_weight: dict[
                int,
                tuple[bool, tuple[int,int,int,int],
                      tuple[float,float,float] | None,
                      tuple[float,float,float] | None]
            ] = {}
            
            # ROI 내부에 있는 밥솥만 매칭 (엄격한 기준 적용)
            for bid in burner_ids:
                if bid not in self._burner_map or not self._burner_map[bid].get("roi"):
                    continue
                rx, ry, rw, rh = self._burner_map[bid]["roi"]
                
                best_dist = float('inf')
                best_body = None
                
                for b in bodies:
                    # 밥솥 박스의 중심점이 캘리브레이션 ROI 내부에 있는지 검사 (약간의 여유 허용)
                    margin_x = rw * 0.2
                    margin_y = rh * 0.2
                    if (rx - margin_x <= b.cx <= rx + rw + margin_x) and \
                       (ry - margin_y <= b.cy <= ry + rh + margin_y):
                        dist = math.hypot(b.cx - (rx + rw/2), b.cy - (ry + rh/2))
                        if dist < best_dist:
                            best_dist = dist
                            best_body = b
                
                if best_body is not None:
                    bb: tuple[float, float, float, float] = (best_body.x1, best_body.y1, best_body.x2, best_body.y2) # type: ignore
                    matched_bodies[bid] = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))

            # ── 딸랑이 독점 매칭: x축 거리 기준 그리디 할당 ──────────────────
            # 각 (bid, weight) 쌍의 x축 거리를 계산한 후 가장 가까운 쌍부터 배정.
            # 한 번 배정된 weight는 다른 pot에 재배정 불가 (독점).
            candidates: list[tuple[float, int, int]] = []  # (x_dist, bid, weight_idx)
            for bid, body_box in matched_bodies.items():
                bx1, by1, bx2, by2 = body_box
                body_cx = (bx1 + bx2) / 2
                w_expansion = (bx2 - bx1) * 0.15
                h_expansion = (by2 - by1) * 0.15
                for wi, w in enumerate(weights):
                    if (bx1 - w_expansion <= w.cx <= bx2 + w_expansion and
                            by1 - h_expansion <= w.cy <= by2 + h_expansion):
                        x_dist = abs(w.cx - body_cx)
                        candidates.append((x_dist, bid, wi))

            candidates.sort(key=lambda t: t[0])
            used_weights: set[int] = set()
            used_bids:    set[int] = set()
            for x_dist, bid, wi in candidates:
                if bid in used_bids or wi in used_weights:
                    continue
                w = weights[wi]
                kp_top: tuple[float, float, float] | None = None
                kp_bot: tuple[float, float, float] | None = None
                if w.keypoints and len(w.keypoints) >= 2:
                    kp_top = w.keypoints[0]
                    kp_bot = w.keypoints[1]
                matched_has_weight[bid] = (
                    True,
                    (int(w.x1), int(w.y1), int(w.x2), int(w.y2)),
                    kp_top, kp_bot,
                )
                used_weights.add(wi)
                used_bids.add(bid)

            # 매칭 안 된 pot은 딸랑이 없음으로 처리
            for bid in matched_bodies:
                if bid not in matched_has_weight:
                    matched_has_weight[bid] = (False, (0, 0, 0, 0), None, None)

            # 매칭 결과로 tracker 업데이트
            for bid in burner_ids:
                if bid in matched_bodies and bid in matched_has_weight:
                    x1, y1, x2, y2 = matched_bodies[bid]
                    has_wt, w_box, kp_top, kp_bot = matched_has_weight[bid]
                    bwx1, bwy1, bwx2, bwy2 = w_box
                    self.last_matched_boxes[bid] = (x1, y1, x2, y2)
                    self._body_ttl[bid] = 15  # 최대 15프레임(약 0.5초) 동안 밥솥 기억
                    if has_wt:
                        self.last_weight_boxes[bid] = w_box
                        if kp_top is not None and kp_bot is not None:
                            self.last_keypoints[bid] = (kp_top, kp_bot)
                        else:
                            self.last_keypoints.pop(bid, None)
                    else:
                        self.last_weight_boxes.pop(bid, None)
                        self.last_keypoints.pop(bid, None)
                        
                    vibrating = self._trackers[bid].update(
                        frame, x1, y1, x2, y2, has_wt, bwx1, bwy1, bwx2, bwy2,
                    )
                    detections[bid] = (True, vibrating)

                    # 무게중심 좌표 저장 (시각화용)
                    c = self._trackers[bid].last_centroid
                    if c is not None:
                        self.last_centroids[bid] = c
                    else:
                        self.last_centroids.pop(bid, None)

                    bsm = self._registry.get(bid)
                    bsm.weight_detected  = has_wt
                    bsm.vibration_score  = self._trackers[bid].score
                    bsm.current_angle    = self._trackers[bid].last_angle
                    bsm.angle_deviation  = self._trackers[bid].last_deviation
                else:
                    # 밥솥을 놓쳤더라도 (수증기 등) body_ttl이 남아있으면 마지막 위치 재사용
                    if self._body_ttl.get(bid, 0) > 0 and bid in self.last_matched_boxes:
                        self._body_ttl[bid] -= 1
                        x1, y1, x2, y2 = self.last_matched_boxes[bid]
                        
                        vibrating = self._trackers[bid].update(frame, x1, y1, x2, y2, False, 0, 0, 0, 0)
                        detections[bid] = (True, vibrating)
                        
                        bsm = self._registry.get(bid)
                        bsm.weight_detected = False
                        bsm.vibration_score = self._trackers[bid].score
                    else:
                        self._trackers[bid].reset()
                        detections[bid] = (False, False)
                        bsm = self._registry.get(bid)
                        bsm.weight_detected = False
                        bsm.vibration_score = 0.0
                        self.last_matched_boxes.pop(bid, None)
                        self.last_weight_boxes.pop(bid, None)
                        self.last_keypoints.pop(bid, None)

        self._registry.update_all(detections)

        # 상태 전환 시 트래커 처리
        for bid in self._trackers:
            cur = self._registry.get(bid).state
            prev = self._prev_states[bid]
            
            # 1. 수동 초기화 또는 밥솥 이탈로 EMPTY가 된 경우 -> 트래커 완전 리셋
            if cur == BurnerState.EMPTY and prev != BurnerState.EMPTY:
                self._trackers[bid].reset()
            
            # 2. 초벌 완료 후 재벌 대기 상태 진입 시 -> 트래커 리셋 (재벌 시작 감지를 위해)
            elif cur == BurnerState.DONE_FIRST and prev == BurnerState.POT_STEAMING_FIRST:
                self._trackers[bid].reset()
                
            self._prev_states[bid] = cur

    def step(self) -> dict[int, np.ndarray]:
        frames = self.read_frames()
        self.detect_and_update()
        return frames
