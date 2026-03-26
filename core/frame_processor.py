"""
프레임 처리기

동작 방식:
  - 전체 프레임에 대해 YOLO 1회 추론 (pot_body / pot_weight 바운딩박스 색출)
  - 캘리브레이션 ROI의 중심점과 YOLO pot_body 중심점을 거리 기반 매칭
  - 독점 매칭: 각 weight를 x축 거리 기준 가장 가까운 pot 하나에만 배정
  - EMA 안정화된 고정 창에서 프레임 간 픽셀 diff를 밥솥 바디 diff로 나눠 진동 판별
    (FrameDiffTracker) — 카메라 흔들림/조명 변화 자동 보정
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT
from core.state_machine import BurnerRegistry, BurnerState


# ── 진동 판별기 (프레임 diff + 카메라 보정) ───────────────────────────────────

class FrameDiffTracker:
    """
    딸랑이 영역의 프레임 간 픽셀 diff를 밥솥 바디 diff로 나눠 진동 판별.

    매 프레임:
      1. YOLO 딸랑이 박스를 EMA로 스무딩 → 고정된 분석 창 위치 확보
      2. 밥솥 바디 박스도 EMA로 스무딩 → 카메라 움직임 기준점
      3. 각 창에서 grayscale 크롭 → 이전 프레임과 절대 diff 평균 계산
      4. diff_ratio = whistle_diff / (pot_diff + 1)
         - 카메라 흔들림: whistle_diff ≈ pot_diff → ratio ≈ 1 → 무시
         - 딸랑이 진동: whistle_diff >> pot_diff → ratio 높음 → 감지

    window 프레임 중 trigger 개 이상 diff_ratio >= threshold → 진동 확정.
    """

    def __init__(self, motion_cfg: dict):
        self._window         = motion_cfg.get("window_frames",        30)
        self._trigger        = motion_cfg.get("trigger_frames",       20)
        self._ratio_thr      = motion_cfg.get("diff_ratio_threshold", 2.0)   # whistle/pot diff 비율 임계값
        self._ema_alpha      = motion_cfg.get("ema_alpha",            0.15)  # EMA 속도 (0~1, 클수록 빠름)

        # EMA 안정화된 딸랑이 창 위치
        self._w_cx: float | None = None
        self._w_cy: float | None = None
        self._w_w:  float | None = None
        self._w_h:  float | None = None

        # EMA 안정화된 밥솥 바디 창 위치
        self._p_cx: float | None = None
        self._p_cy: float | None = None
        self._p_w:  float | None = None
        self._p_h:  float | None = None

        # 이전 프레임 그레이스케일 크롭 (diff 계산용)
        self._prev_w: np.ndarray | None = None
        self._prev_p: np.ndarray | None = None

        self._cv_hist: deque[bool] = deque(maxlen=self._window)
        self._last_ratio:    float = 0.0   # 디버그: 마지막 diff ratio
        self._last_centroid: tuple[int, int] | None = None  # 시각화: EMA 딸랑이 중심

    def reset(self) -> None:
        self._w_cx = self._w_cy = self._w_w = self._w_h = None
        self._p_cx = self._p_cy = self._p_w = self._p_h = None
        self._prev_w = self._prev_p = None
        self._cv_hist.clear()
        self._last_ratio    = 0.0
        self._last_centroid = None

    @property
    def last_angle(self) -> float | None:
        """UI 호환용 — diff ratio 반환"""
        return self._last_ratio if self._last_ratio > 0 else None

    @property
    def last_deviation(self) -> float:
        """UI 호환용 — diff ratio 반환"""
        return self._last_ratio

    @property
    def last_centroid(self) -> tuple[int, int] | None:
        """시각화용 — EMA 안정화된 딸랑이 중심"""
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
    def _crop_gray(frame: np.ndarray, cx: float, cy: float, w: float, h: float) -> np.ndarray | None:
        """EMA 중심 기준 고정 크기 grayscale 크롭"""
        x1 = int(round(cx - w / 2))
        y1 = int(round(cy - h / 2))
        x2 = x1 + max(1, int(round(w)))
        y2 = y1 + max(1, int(round(h)))
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    def update(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,   # pot_body 박스
        has_weight: bool = False,
        wx1: int = 0, wy1: int = 0, wx2: int = 0, wy2: int = 0,
    ) -> bool:
        motion = False
        a = self._ema_alpha

        if has_weight and wx2 > wx1 and wy2 > wy1:
            # 딸랑이 EMA 위치 업데이트 (내부 60% 영역)
            self._w_cx = self._ema(self._w_cx, (wx1 + wx2) / 2,       a)
            self._w_cy = self._ema(self._w_cy, (wy1 + wy2) / 2,       a)
            self._w_w  = self._ema(self._w_w,  (wx2 - wx1) * 0.8,     a)
            self._w_h  = self._ema(self._w_h,  (wy2 - wy1) * 0.8,     a)

            # 밥솥 바디 EMA 위치 업데이트 (중앙 40% 영역)
            self._p_cx = self._ema(self._p_cx, (x1 + x2) / 2,         a)
            self._p_cy = self._ema(self._p_cy, (y1 + y2) / 2,         a)
            self._p_w  = self._ema(self._p_w,  (x2 - x1) * 0.5,       a)
            self._p_h  = self._ema(self._p_h,  (y2 - y1) * 0.5,       a)

            self._last_centroid = (int(self._w_cx), int(self._w_cy))

            w_gray = self._crop_gray(frame, self._w_cx, self._w_cy, self._w_w, self._w_h)
            p_gray = self._crop_gray(frame, self._p_cx, self._p_cy, self._p_w, self._p_h)

            if (w_gray is not None and p_gray is not None and
                    self._prev_w is not None and self._prev_p is not None and
                    self._prev_w.shape == w_gray.shape and
                    self._prev_p.shape == p_gray.shape):

                w_diff = float(np.mean(np.abs(w_gray.astype(np.int16) - self._prev_w.astype(np.int16))))
                p_diff = float(np.mean(np.abs(p_gray.astype(np.int16) - self._prev_p.astype(np.int16))))

                self._last_ratio = w_diff / (p_diff + 1.0)  # +1: 0 나눗셈 방지
                if self._last_ratio >= self._ratio_thr:
                    motion = True

            if w_gray is not None:
                self._prev_w = w_gray
            if p_gray is not None:
                self._prev_p = p_gray
        else:
            # 딸랑이 미감지: 이전 프레임 초기화 (박스 위치 변화로 인한 false diff 방지)
            self._prev_w = None
            self._prev_p = None

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
