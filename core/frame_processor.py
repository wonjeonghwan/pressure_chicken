"""
프레임 처리기

하이브리드 모드 (옵션 1 적용):
  - 전체 프레임에 대해 YOLO 1회 추론 (pot_body 색출)
  - 캘리브레이션 ROI의 '중심점(Anchor)'과 YOLO pot_body 중심점을 거리 기반 매칭
  - 매칭된 밥솥의 상단 절반(Top-half) 영역 크롭 -> EMA(지수이동평균)로 흔들림 보정
  - 픽셀 차이(Pixel Diff)를 통해 증기/딸랑이 모션 감지 (VibrationTracker)
"""

from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT
from core.state_machine import BurnerRegistry, BurnerState


# ── 진동 판별기 (하이브리드 픽셀 모션) ──────────────────────────────────────

class HybridVibrationTracker:
    """
    YOLO가 도출한 밥솥 바디(pot_body)와 딸랑이(pot_weight)의
    정중앙점(Center) 좌표가 최근 N 프레임 동안 얼마나 분산(Variance)되었는지를 측정하여 진동을 판별한다.
    광학 흐름(Optical Flow) 방식의 저프레임 추적 실패 및 노이즈 문제를 해결한 방식(Plan A).
    """

    def __init__(self, motion_cfg: dict):
        self._window = motion_cfg.get("window_frames", 30)
        self._trigger = motion_cfg.get("trigger_frames", 20)
        self._min_w_activity = motion_cfg.get("rel_x_threshold", 3.0)
        self._body_ratio_mul = motion_cfg.get("body_ratio_multiplier", 1.5)
        self._motion_ratio = motion_cfg.get("min_motion_ratio", 0.15)
        self._threshold = motion_cfg.get("threshold", 15)
        self._alpha = 0.1
        
        self._frame_n: int = 0
        self._cv_hist: deque[bool] = deque(maxlen=self._window)
        self._weight_ttl = 0
        
        # 최근 좌표 기록용 Deque (최근 15프레임 정도의 궤적을 비교)
        self._history_len = min(15, self._window)
        self._w_boxes: deque[tuple[float, float, float, float]] = deque(maxlen=self._history_len)
        self._b_boxes: deque[tuple[float, float, float, float]] = deque(maxlen=self._history_len)
        
        # EMA(지수이동평균) 중심 박스 및 픽셀 차분용 저장소
        self._ema_w_box: tuple[float, float, float, float] | None = None
        self._prev_roi_gray: np.ndarray | None = None
        
        self.reset()

    def reset(self) -> None:
        self._frame_n  = 0
        self._cv_hist.clear()
        self._w_boxes.clear()
        self._b_boxes.clear()
        self._weight_ttl = 0
        self._ema_w_box = None
        self._prev_roi_gray = None

    def update(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
               has_weight: bool = False,
               wx1: int = 0, wy1: int = 0, wx2: int = 0, wy2: int = 0) -> bool:
        self._frame_n += 1
        
        if has_weight:
            self._weight_ttl = 60 # 약 2초간 딸랑이 본 기억 유지
        else:
            self._weight_ttl = max(0, self._weight_ttl - 1)
            
        motion = False
        
        # 바디 전체 박스 기록
        self._b_boxes.append((float(x1), float(y1), float(x2), float(y2)))
        
        if has_weight and wx2 > wx1 and wy2 > wy1:
            self._w_boxes.append((float(wx1), float(wy1), float(wx2), float(wy2)))
            
            # 충분한 좌표 데이터가 쌓였을 때 픽셀 차분 계산
            if len(self._w_boxes) >= 5 and len(self._b_boxes) >= 5:
                # 진동 판별 (선택지 A): YOLO 바운딩 박스 분산 계산 버림, 오직 EMA 기반 내부 픽셀 텍스처 변화만 신뢰
                pixel_motion = False
                
                # EMA(지수이동평균) 고정 눈동자 업데이트
                if self._ema_w_box is None:
                    self._ema_w_box = (float(wx1), float(wy1), float(wx2), float(wy2))
                else:
                    ex1, ey1, ex2, ey2 = self._ema_w_box
                    self._ema_w_box = (
                        ex1 * (1 - self._alpha) + wx1 * self._alpha,
                        ey1 * (1 - self._alpha) + wy1 * self._alpha,
                        ex2 * (1 - self._alpha) + wx2 * self._alpha,
                        ey2 * (1 - self._alpha) + wy2 * self._alpha
                    )
                
                assert self._ema_w_box is not None

                # EMA 렌즈를 기준으로 가장자리를 제외한 중앙 80% 영역 추출
                ex1, ey1, ex2, ey2 = self._ema_w_box
                ew = ex2 - ex1
                eh = ey2 - ey1
                
                if ew > 5 and eh > 5:
                    cx1 = int(ex1 + ew * 0.1)
                    cy1 = int(ey1 + eh * 0.1)
                    cx2 = int(ex2 - ew * 0.1)
                    cy2 = int(ey2 - eh * 0.1)
                    
                    fh, fw = frame.shape[:2]
                    cx1, cy1 = max(0, cx1), max(0, cy1)
                    cx2, cy2 = min(fw, cx2), min(fh, cy2)
                    
                    if cx2 > cx1 and cy2 > cy1:
                        roi = frame[cy1:cy2, cx1:cx2]
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        # 미세한 크기 변동 오차를 없애기 위해 32x32로 고정
                        gray_roi = cv2.resize(gray_roi, (32, 32))
                        
                        if self._prev_roi_gray is not None:
                            diff = cv2.absdiff(self._prev_roi_gray, gray_roi)
                            _, thresh = cv2.threshold(diff, self._threshold, 255, cv2.THRESH_BINARY)
                            changed_pixels = cv2.countNonZero(thresh)
                            changed_ratio = changed_pixels / (32 * 32)
                            
                            if changed_ratio >= self._motion_ratio:
                                pixel_motion = True
                                
                        self._prev_roi_gray = gray_roi

                # 오직 내부 속살 픽셀이 변했을 때만 진짜 진동으로 판정 (선택지 A - 픽셀 올인)
                if pixel_motion:
                    motion = True
        
        self._cv_hist.append(motion)
        return self._check()

    def _check(self) -> bool:
        hist = list(self._cv_hist)
        if len(hist) < self._window:
            return False
        return sum(hist) >= self._trigger

    @property
    def score(self) -> float:
        if not self._cv_hist:
            return 0.0
        return min(1.0, sum(self._cv_hist) / self._trigger)


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

        # 화구별 HybridVibrationTracker
        self._trackers: dict[int, HybridVibrationTracker] = {
            b["id"]: HybridVibrationTracker(motion_cfg)
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
        self.last_matched_boxes: dict[int, tuple[int, int, int, int]] = {}
        self.last_weight_boxes:  dict[int, tuple[int, int, int, int]] = {}

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
                    d.x1 += cx1
                    d.x2 += cx1
                    d.y1 += cy1
                    d.y2 += cy1
            else:
                dets = self._detector.detect(frame)
            
            bodies = [d for d in dets if d.class_id == CLASS_POT_BODY and d.confidence >= 0.3]
            weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]
            
            matched_bodies: dict[int, tuple[int, int, int, int]] = {}
            matched_has_weight: dict[int, tuple[bool, tuple[int, int, int, int]]] = {}
            
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
                    
                    # 상하반전(거꾸로 찰영) 대비: 밥솥 바운딩 박스를 기준으로 모든 전/후/좌/우 바운더리를 패딩으로 확장
                    has_w = False
                    w_box = (0, 0, 0, 0)
                    w_expansion = (bb[2] - bb[0]) * 0.15  # 가로 15% 확장
                    h_expansion = (bb[3] - bb[1]) * 0.15  # 세로 15% 확장
                    
                    for w in weights:
                        wx, wy = w.cx, w.cy # type: ignore
                        if bb[0] - w_expansion <= wx <= bb[2] + w_expansion:
                            # 밥솥 위축(Top) 절반이라는 조건 해제 (거꾸로 찍히면 Bottom에 있으므로)
                            # 상하 15% 여유 범위 안에 위치한 모든 딸랑이 탐지
                            if bb[1] - h_expansion <= wy <= bb[3] + h_expansion:
                                has_w = True
                                w_box = (int(w.x1), int(w.y1), int(w.x2), int(w.y2)) # type: ignore
                                break
                    matched_has_weight[bid] = (has_w, w_box)

            # 매칭 결과로 tracker 업데이트
            for bid in burner_ids:
                if bid in matched_bodies and bid in matched_has_weight:
                    x1, y1, x2, y2 = matched_bodies[bid]
                    has_wt, w_box = matched_has_weight[bid]
                    bwx1, bwy1, bwx2, bwy2 = w_box
                    self.last_matched_boxes[bid] = (x1, y1, x2, y2)
                    self._body_ttl[bid] = 30  # 최대 30프레임(약 1초) 동안 밥솥 기억
                    if has_wt:
                        self.last_weight_boxes[bid] = w_box
                    else:
                        self.last_weight_boxes.pop(bid, None)
                        
                    vibrating = self._trackers[bid].update(frame, x1, y1, x2, y2, has_wt, bwx1, bwy1, bwx2, bwy2)
                    detections[bid] = (True, vibrating)
                    
                    bsm = self._registry.get(bid)
                    bsm.weight_detected = has_wt  # 딸랑이 물체가 보이면 즉시 불 켜기
                    bsm.vibration_score = self._trackers[bid].score
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
