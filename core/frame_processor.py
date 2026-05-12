"""
프레임 처리기

Phase 1 (Stabilizer) + Phase 2 (OpticalFlow) 통합 파이프라인:

  1. Phase 1 — Stabilizer: 소스별 카메라 흔들림 보정
     - Grid 기반 LK tracking → RANSAC → EMA warpAffine
     - YOLO 추론 전에 적용 → 감지 안정성 향상

  2. YOLO-seg 추론: pot_body / pot_weight 감지
     - 보정된 프레임으로 추론 → 박스/마스크 좌표는 보정 좌표계

  3. 매칭: 캘리브레이션 ROI → pot_body → pot_weight 독점 매칭

  4. Phase 2 — OpticalFlowDetector: 화구별 움직임 판별
     - 보정된 프레임 + 매칭된 w_box → Farneback dense flow RMS
     - EMA 평활화로 YOLO bbox jitter 스파이크 억제
     - window/trigger 방식으로 지속 움직임 확정
"""

from __future__ import annotations

import math

import numpy as np

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT
from core.state_machine import BurnerRegistry, BurnerState
from core.stabilizer import Stabilizer
from core.optical_flow import OpticalFlowDetector


class FrameProcessor:
    """
    VideoSource 들에서 프레임을 읽고 Phase1+2 파이프라인으로 감지 → 상태머신 갱신.
    """

    def __init__(
        self,
        sources:     dict[int, VideoSource],
        burner_cfgs: list[dict],
        registry:    BurnerRegistry,
        detector:    BurnerDetector,
        config:      dict,
    ):
        self._sources    = sources
        self._registry   = registry
        self._detector   = detector
        self._burner_map = {b["id"]: b for b in burner_cfgs}

        stab_cfg = config.get("stabilizer", {})
        flow_cfg = config.get("optical_flow", {})

        # Phase 1: 소스별 Stabilizer
        self._stabilizers: dict[int, Stabilizer] = {
            sc["id"]: Stabilizer(stab_cfg)
            for sc in config.get("sources", [])
        }

        # Phase 2: 화구별 OpticalFlowDetector
        self._oflow: dict[int, OpticalFlowDetector] = {
            b["id"]: OpticalFlowDetector(flow_cfg)
            for b in burner_cfgs
        }



        # 화구별 캘리브레이션 앵커 포인트
        self._anchors: dict[int, tuple[int, int]] = {}
        for b in burner_cfgs:
            roi = b.get("roi")
            if roi:
                x, y, w, h = roi
                self._anchors[b["id"]] = (x + w // 2, y + h // 2)

        self._frame_cache:      dict[int, np.ndarray | None] = {}
        self._stabilized_cache: dict[int, np.ndarray | None] = {}
        self._prev_states: dict[int, BurnerState] = {
            b["id"]: BurnerState.EMPTY for b in burner_cfgs
        }
        self._body_ttl: dict[int, int] = {b["id"]: 0 for b in burner_cfgs}

        # UI 오버레이용 마지막 감지 결과
        self.last_matched_boxes: dict[int, tuple[int, int, int, int]] = {}
        self.last_weight_boxes:  dict[int, tuple[int, int, int, int]] = {}
        self.last_centroids:     dict[int, tuple[int, int]] = {}
        self.last_mask_xys:      dict[int, np.ndarray] = {}

    def oflow(self, bid: int) -> "OpticalFlowDetector | None":
        return self._oflow.get(bid)

    def read_frames(self) -> dict[int, np.ndarray]:
        frames: dict[int, np.ndarray | None] = {}
        for src_id, src in self._sources.items():
            ret, frame = src.read()
            frames[src_id] = frame if ret else None
        self._frame_cache = frames
        return {sid: f for sid, f in frames.items() if f is not None}

    def detect_and_update(self) -> None:
        # 수동 초기화(EMPTY 전환) 감지 → 해당 화구 상태 초기화
        for bid in self._burner_map:
            bsm = self._registry.get(bid)
            if bsm.state == BurnerState.EMPTY and self._prev_states[bid] != BurnerState.EMPTY:
                self._oflow[bid].reset()
                self._prev_states[bid] = BurnerState.EMPTY
                self.last_matched_boxes.pop(bid, None)
                self.last_weight_boxes.pop(bid, None)
                self.last_mask_xys.pop(bid, None)
                self._body_ttl[bid] = 0

        frames = self._frame_cache
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

            # ── Phase 1: 카메라 흔들림 보정 ──────────────────────────────
            stab = self._stabilizers.get(src_id)
            stabilized = stab.stabilize(frame) if stab else frame
            self._stabilized_cache[src_id] = stabilized

            # ── YOLO 추론 (보정된 프레임) ────────────────────────────────
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0.0, 0.0
            has_roi = False
            for bid in burner_ids:
                roi = self._burner_map[bid].get("roi")
                if roi:
                    rx, ry, rw, rh = roi
                    min_x = min(min_x, rx)
                    min_y = min(min_y, ry)
                    max_x = max(max_x, rx + rw)
                    max_y = max(max_y, ry + rh)
                    has_roi = True

            if has_roi:
                margin = 50
                fh, fw = stabilized.shape[:2]
                cx1 = max(0, int(min_x) - margin)
                cy1 = max(0, int(min_y) - margin)
                cx2 = min(fw, int(max_x) + margin)
                cy2 = min(fh, int(max_y) + margin)
                crop = stabilized[cy1:cy2, cx1:cx2]
                dets = self._detector.detect(crop)
                for d in dets:
                    d.x1 += cx1; d.x2 += cx1
                    d.y1 += cy1; d.y2 += cy1
                    if d.mask_xy is not None:
                        d.mask_xy = d.mask_xy + np.array([[cx1, cy1]], dtype=np.float32)
            else:
                dets = self._detector.detect(stabilized)

            bodies  = [d for d in dets if d.class_id == CLASS_POT_BODY   and d.confidence >= 0.3]
            weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]

            # ── 밥솥 매칭 (ROI 기반) ─────────────────────────────────────
            matched_bodies: dict[int, tuple[int, int, int, int]] = {}
            for bid in burner_ids:
                roi = self._burner_map[bid].get("roi")
                if not roi:
                    continue
                rx, ry, rw, rh = roi
                best_dist, best_body = float('inf'), None
                for b in bodies:
                    mx, my = rw * 0.2, rh * 0.2
                    if (rx - mx <= b.cx <= rx + rw + mx) and (ry - my <= b.cy <= ry + rh + my):
                        dist = math.hypot(b.cx - (rx + rw / 2), b.cy - (ry + rh / 2))
                        if dist < best_dist:
                            best_dist, best_body = dist, b
                if best_body is not None:
                    matched_bodies[bid] = (int(best_body.x1), int(best_body.y1),
                                           int(best_body.x2), int(best_body.y2))

            # ── 딸랑이 독점 매칭 (x축 거리 그리디) ──────────────────────
            matched_has_weight: dict[int, tuple[bool, tuple, np.ndarray | None]] = {}
            candidates: list[tuple[float, int, int]] = []
            for bid, body_box in matched_bodies.items():
                bx1, by1, bx2, by2 = body_box
                bcx = (bx1 + bx2) / 2
                we = (bx2 - bx1) * 0.15
                he = (by2 - by1) * 0.15
                for wi, w in enumerate(weights):
                    if bx1 - we <= w.cx <= bx2 + we and by1 - he <= w.cy <= by2 + he:
                        candidates.append((abs(w.cx - bcx), bid, wi))

            candidates.sort(key=lambda t: t[0])
            used_weights: set[int] = set()
            used_bids:    set[int] = set()
            for _, bid, wi in candidates:
                if bid in used_bids or wi in used_weights:
                    continue
                w = weights[wi]
                matched_has_weight[bid] = (
                    True,
                    (int(w.x1), int(w.y1), int(w.x2), int(w.y2)),
                    w.mask_xy,
                )
                used_weights.add(wi)
                used_bids.add(bid)

            for bid in matched_bodies:
                if bid not in matched_has_weight:
                    matched_has_weight[bid] = (False, (0, 0, 0, 0), None)

            # ── Phase 2: optical flow 움직임 판별 ────────────────────────
            for bid in burner_ids:
                if bid in matched_bodies and bid in matched_has_weight:
                    x1, y1, x2, y2 = matched_bodies[bid]
                    has_wt, w_box, mask_xy = matched_has_weight[bid]
                    self.last_matched_boxes[bid] = (x1, y1, x2, y2)
                    self._body_ttl[bid] = 15

                    if has_wt:
                        self.last_weight_boxes[bid] = w_box
                        cx = (w_box[0] + w_box[2]) // 2
                        cy = (w_box[1] + w_box[3]) // 2
                        self.last_centroids[bid] = (cx, cy)
                        if mask_xy is not None:
                            self.last_mask_xys[bid] = mask_xy
                        else:
                            self.last_mask_xys.pop(bid, None)
                    else:
                        self.last_weight_boxes.pop(bid, None)
                        self.last_centroids.pop(bid, None)
                        self.last_mask_xys.pop(bid, None)

                    oflow_box  = w_box    if has_wt else None
                    oflow_mask = mask_xy  if has_wt else None
                    vibrating_p2, _ = self._oflow[bid].update(stabilized, oflow_box, oflow_mask)

                    # 최종 진동 판정 (Phase 2 단독 수행)
                    vibrating = vibrating_p2

                    detections[bid] = (True, vibrating)

                    bsm = self._registry.get(bid)
                    bsm.weight_detected = has_wt
                    bsm.vibration_score = self._oflow[bid].score
                    # current_angle 에 smoothed RMS 저장 (UI 표시용)
                    bsm.current_angle   = self._oflow[bid].last_smoothed_rms if has_wt else None
                    bsm.angle_deviation = self._oflow[bid].last_normalized_rms

                else:
                    if self._body_ttl.get(bid, 0) > 0 and bid in self.last_matched_boxes:
                        self._body_ttl[bid] -= 1
                        vibrating, _ = self._oflow[bid].update(stabilized, None)
                        detections[bid] = (True, vibrating)
                        bsm = self._registry.get(bid)
                        bsm.weight_detected = False
                        bsm.vibration_score = self._oflow[bid].score
                    else:
                        self._oflow[bid].reset()
                        detections[bid] = (False, False)
                        bsm = self._registry.get(bid)
                        bsm.weight_detected = False
                        bsm.vibration_score = 0.0
                        self.last_matched_boxes.pop(bid, None)
                        self.last_weight_boxes.pop(bid, None)
                        self.last_mask_xys.pop(bid, None)

        self._registry.update_all(detections)

        # 상태 전환 후처리
        for bid in self._oflow:
            cur  = self._registry.get(bid).state
            prev = self._prev_states[bid]

            if cur == BurnerState.EMPTY and prev != BurnerState.EMPTY:
                self._oflow[bid].reset()

            elif cur == BurnerState.DONE_FIRST and prev == BurnerState.POT_STEAMING_FIRST:
                self._oflow[bid].reset()

            elif cur == BurnerState.WAIT_SECOND and prev == BurnerState.DONE_FIRST:
                # 재벌대기 진입 시 flow 히스토리 초기화 — 이전 DONE_FIRST 구간 누적 제거
                self._oflow[bid].reset()

            self._prev_states[bid] = cur

    def step(self) -> dict[int, np.ndarray]:
        frames = self.read_frames()
        self.detect_and_update()
        return frames
