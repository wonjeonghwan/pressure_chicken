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
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT, CLASS_VENT
from core.state_machine import BurnerRegistry, BurnerState
from core.stabilizer import Stabilizer
from core.optical_flow import OpticalFlowDetector


def _merged_cfg(global_cfg: dict, *overlays: dict) -> dict:
    """global default → overlay 순서로 dict를 병합. 뒤 overlay가 앞 값을 덮어씀.

    각 overlay는 부분 dict만 가질 수 있음 (override 하고 싶은 키만 명시).
    """
    result = dict(global_cfg)
    for ov in overlays:
        if ov:
            result.update(ov)
    return result


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

        global_stab = config.get("stabilizer", {})
        global_flow = config.get("optical_flow", {})
        self._global_flow = global_flow

        # Phase 1: 소스별 Stabilizer — source.stabilizer 가 있으면 global을 부분 덮어씀
        self._stabilizers: dict[int, Stabilizer] = {}
        for sc in config.get("sources", []):
            merged = _merged_cfg(global_stab, sc.get("stabilizer", {}))
            self._stabilizers[sc["id"]] = Stabilizer(merged)

        # Phase 2: 화구별 OpticalFlowDetector
        # 합성 우선순위: global < source.optical_flow < burner.optical_flow (rms_override_enabled=False면 burner 단계 생략)
        # 카메라 모델/해상도 다르면 source별, 특정 화구만 다르면 burner별 override 가능
        self._sources_by_id = {sc["id"]: sc for sc in config.get("sources", [])}
        self._oflow: dict[int, OpticalFlowDetector] = {}
        for b in burner_cfgs:
            sc = self._sources_by_id.get(b.get("source_id"), {})
            merged = _merged_cfg(
                global_flow,
                sc.get("optical_flow", {}),
                self._burner_flow_override(b),
            )
            self._oflow[b["id"]] = OpticalFlowDetector(merged)



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
        # Dev 모드 오버레이 + 로그용 — 화구 window 내 vent(class=2) 검출 박스 (매칭 승패와 무관하게 전부 기록)
        self.last_vent_boxes:    dict[int, list[tuple[int, int, int, int]]] = {}
        # 로그 비교용 — 화구 window 내 weight(class=1) 검출 박스 전부 (dedup 이전, 중복/오분류 위치 분석용)
        self.last_weight_candidate_boxes: dict[int, list[tuple[int, int, int, int]]] = {}

    def oflow(self, bid: int) -> "OpticalFlowDetector | None":
        return self._oflow.get(bid)

    # ── 화구별 rms_threshold 단독(override) 설정 — 개발자 모드 UI용 ──────────
    def _burner_flow_override(self, b: dict) -> dict:
        """optical_flow.rms_threshold가 설정된 화구만 병합에 반영 (설정 여부 자체가 활성화 신호)."""
        return b.get("optical_flow", {}) if "rms_threshold" in b.get("optical_flow", {}) else {}

    def get_global_rms_threshold(self) -> float:
        return self._global_flow.get("rms_threshold", 0.5)

    def get_own_rms_threshold(self, bid: int) -> float:
        """화구 단독 override 값 (사용 여부와 무관하게 저장된 수치). 없으면 현재 유효값을 기본으로."""
        b = self._burner_map.get(bid)
        if b is None:
            return self.get_global_rms_threshold()
        own = b.get("optical_flow", {}).get("rms_threshold")
        if own is not None:
            return own
        oflow = self._oflow.get(bid)
        return oflow.rms_threshold if oflow else self.get_global_rms_threshold()

    def adjust_own_rms_threshold(self, bid: int, delta: float) -> None:
        """화구 단독 rms_threshold를 delta만큼 조정 (사용 여부와 무관하게 저장값 자체를 변경)."""
        b = self._burner_map.get(bid)
        if b is None:
            return
        flow_cfg = b.setdefault("optical_flow", {})
        current = flow_cfg.get("rms_threshold", self.get_own_rms_threshold(bid))
        flow_cfg["rms_threshold"] = round(max(0.0, min(2.0, current + delta)), 2)
        self._apply_rms_threshold(bid)

    def _apply_rms_threshold(self, bid: int) -> None:
        """burner_map의 현재 override 설정을 실행 중인 OpticalFlowDetector에 즉시 반영."""
        b = self._burner_map.get(bid)
        oflow = self._oflow.get(bid)
        if b is None or oflow is None:
            return
        sc = self._sources_by_id.get(b.get("source_id"), {})
        merged = _merged_cfg(self._global_flow, sc.get("optical_flow", {}), self._burner_flow_override(b))
        oflow.rms_threshold = merged.get("rms_threshold", oflow.rms_threshold)

    def estimate_roi_coverage(self, source_resolution: tuple[int, int] | None = None) -> dict[int, dict]:
        """카메라별 ROI 합집합 면적을 비율(0~1)로 산출.

        ROI들이 겹칠 수 있으니 픽셀 마스크로 정확히 계산.
        source_resolution을 모르면 ROI bounding box 면적으로 근사.

        Returns:
            { source_id: {"n_burners": int, "coverage": float (0~1) or None, "bbox_w_h": (W,H) or None} }
        """
        per_src: dict[int, list[tuple[int, int, int, int]]] = {}
        for b in self._burner_map.values():
            roi = b.get("roi")
            if not roi:
                continue
            sid = b.get("source_id", 0)
            per_src.setdefault(sid, []).append(tuple(roi))

        out: dict[int, dict] = {}
        for sid, rois in per_src.items():
            if not rois:
                out[sid] = {"n_burners": 0, "coverage": None, "bbox_w_h": None}
                continue
            res = source_resolution
            if res is None:
                # ROI bounding box를 화면으로 가정 (보수적 추정 — 실제 coverage는 ≤ 이 값)
                max_x = max(r[0] + r[2] for r in rois)
                max_y = max(r[1] + r[3] for r in rois)
                res = (max_x, max_y)
            W, H = res
            if W <= 0 or H <= 0:
                out[sid] = {"n_burners": len(rois), "coverage": None, "bbox_w_h": (W, H)}
                continue
            mask = np.zeros((H, W), dtype=np.uint8)
            for x, y, w, h in rois:
                x1 = max(0, x); y1 = max(0, y)
                x2 = min(W, x + w); y2 = min(H, y + h)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 1
            covered = int(mask.sum())
            out[sid] = {
                "n_burners": len(rois),
                "coverage": covered / (W * H),
                "bbox_w_h": (W, H),
            }
        return out

    def read_frames(self) -> dict[int, np.ndarray]:
        """각 소스에서 최신 프레임 수집.

        카메라 소스: get_latest() — 캡처 스레드 버퍼에서 비블로킹 조회.
        파일 소스:   read()      — 기존 동기 방식 유지 (frame skip 타이밍 보존).
        """
        frames: dict[int, np.ndarray | None] = {}
        for src_id, src in self._sources.items():
            ret, frame = src.get_latest()
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
                self.last_vent_boxes.pop(bid, None)
                self.last_weight_candidate_boxes.pop(bid, None)
                bsm.vent_count = 0
                bsm.weight_class_count = 0
                self._body_ttl[bid] = 0

        frames = self._frame_cache
        detections: dict[int, tuple[bool, bool]] = {}

        src_burners: dict[int, list[int]] = {}
        for bid, cfg in self._burner_map.items():
            src_burners.setdefault(cfg["source_id"], []).append(bid)

        for src_id, burner_ids in src_burners.items():
            frame = frames.get(src_id)

            if frame is None:
                # 카메라 오프라인 → pot_absent_count 누적을 막으면서 타이머는 정상 진행
                # EMPTY: (False,False) → EMPTY 유지
                # 그 외: (True,False) → 현재 상태 보존 + STEAMING 타이머 완료 시 정상 전환
                for bid in burner_ids:
                    if self._registry.get(bid).state == BurnerState.EMPTY:
                        detections[bid] = (False, False)
                    else:
                        detections[bid] = (True, False)
                continue

            # ── Phase 1: 카메라 흔들림 보정 ──────────────────────────────
            stab = self._stabilizers.get(src_id)
            stabilized = stab.stabilize(frame) if stab else frame
            self._stabilized_cache[src_id] = stabilized

            # ── YOLO 추론 (보정된 프레임) ────────────────────────────────
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0.0, 0.0
            min_short_side = float('inf')   # 동적 margin 계산용
            has_roi = False
            for bid in burner_ids:
                roi = self._burner_map[bid].get("roi")
                if roi:
                    rx, ry, rw, rh = roi
                    min_x = min(min_x, rx)
                    min_y = min(min_y, ry)
                    max_x = max(max_x, rx + rw)
                    max_y = max(max_y, ry + rh)
                    min_short_side = min(min_short_side, min(rw, rh))
                    has_roi = True

            if has_roi:
                # 동적 margin: ROI 짧은 변의 10%, 안전선 [30, 50]
                #  - 작은 ROI → 작은 margin → YOLO crop 면적 절감
                #  - 큰 ROI → 큰 margin → stabilizer 보정 후 객체 이탈 방지
                margin = max(30, min(50, int(min_short_side * 0.1)))
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

            bodies  = [d for d in dets if d.class_id == CLASS_POT_BODY]
            weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT]
            vents   = [d for d in dets if d.class_id == CLASS_VENT]

            # ── 밥솥 매칭 (ROI 기반, 독점 그리디) ───────────────────────
            # 가장 가까운 화구가 body를 독점 — 인접 화구 간섭 방지
            matched_bodies: dict[int, tuple[int, int, int, int]] = {}
            body_candidates: list[tuple[float, int, int]] = []  # (dist, bid, body_idx)
            for bid in burner_ids:
                roi = self._burner_map[bid].get("roi")
                if not roi:
                    continue
                rx, ry, rw, rh = roi
                for bi, b in enumerate(bodies):
                    if (rx <= b.cx <= rx + rw) and (ry <= b.cy <= ry + rh):
                        dist = math.hypot(b.cx - (rx + rw / 2), b.cy - (ry + rh / 2))
                        body_candidates.append((dist, bid, bi))

            body_candidates.sort(key=lambda t: t[0])
            used_bodies: set[int] = set()
            used_body_bids: set[int] = set()
            for dist, bid, bi in body_candidates:
                if bid in used_body_bids or bi in used_bodies:
                    continue
                b = bodies[bi]
                matched_bodies[bid] = (int(b.x1), int(b.y1), int(b.x2), int(b.y2))
                used_bodies.add(bi)
                used_body_bids.add(bid)

            # ── 딸랑이 독점 매칭 (weight+vent 통합 후보, 면적 큰 순 그리디) ──
            # YOLO가 vent(class=2)를 weight(class=1)로, 또는 그 반대로 오분류하는
            # 경우가 있어 weight-class만 신뢰하지 않고 vent-class도 후보 풀에 포함.
            # 실측 라벨 면적 중앙값이 weight(0.0007) > vent(0.00034)로 대략 2배 —
            # 후보가 여럿(애매)이거나 weight-class가 아예 없을 때도 "더 큰 쪽이
            # 진짜 딸랑이일 가능성이 높다"는 도메인 사전지식으로 모델의 class 출력을
            # 재검증한다 (면적 내림차순 그리디 배정이므로 애매하지 않은 단독 후보는
            # 기존과 동일하게 그대로 선택됨).
            weight_like = weights + vents
            matched_has_weight: dict[int, tuple[bool, tuple, np.ndarray | None]] = {}
            candidates: list[tuple[float, int, int]] = []
            for bid, body_box in matched_bodies.items():
                bx1, by1, bx2, by2 = body_box
                we = (bx2 - bx1) * 0.15
                he = (by2 - by1) * 0.15
                wx1, wy1, wx2, wy2 = bx1 - we, by1 - he, bx2 + we, by2 + he
                # body 검출이 비정상적으로 크거나 어긋나면 ±15% window가 자기 캘리브레이션
                # roi를 벗어나 옆 화구 영역까지 침범할 수 있음 (실측: 13번 화구가 11번의
                # vent를 자기 weight로 채간 사례) — roi와 교집합으로 클리핑해 화구 간 오염 방지.
                roi = self._burner_map[bid].get("roi")
                if roi:
                    rx, ry, rw, rh = roi
                    wx1 = max(wx1, rx)
                    wy1 = max(wy1, ry)
                    wx2 = min(wx2, rx + rw)
                    wy2 = min(wy2, ry + rh)
                vent_boxes_here:   list[tuple[int, int, int, int]] = []
                weight_boxes_here: list[tuple[int, int, int, int]] = []
                for wi, w in enumerate(weight_like):
                    if wx1 <= w.cx <= wx2 and wy1 <= w.cy <= wy2:
                        area = (w.x2 - w.x1) * (w.y2 - w.y1)
                        candidates.append((area, bid, wi))
                        box = (int(w.x1), int(w.y1), int(w.x2), int(w.y2))
                        if w.class_id == CLASS_VENT:
                            vent_boxes_here.append(box)
                        else:
                            weight_boxes_here.append(box)
                # Dev 모드 오버레이 + 로그용 — 매칭 승패와 무관하게 window 내 검출을 위치까지 전부 기록
                # (weight-class 후보가 여럿일 때 같은 물체의 중복 검출인지, 서로 다른 위치의
                #  별개 물체(예: vent 오분류)인지 좌표 비교로 구분할 수 있게 함)
                if vent_boxes_here:
                    self.last_vent_boxes[bid] = vent_boxes_here
                else:
                    self.last_vent_boxes.pop(bid, None)
                if weight_boxes_here:
                    self.last_weight_candidate_boxes[bid] = weight_boxes_here
                else:
                    self.last_weight_candidate_boxes.pop(bid, None)
                bsm = self._registry.get(bid)
                bsm.vent_count         = len(vent_boxes_here)
                bsm.weight_class_count = len(weight_boxes_here)

            # 화구 하나당 후보(weight-class + vent-class 통째로)는 면적이 가장 큰 딱 하나만
            # 채택되고 나머지는 전부 탈락한다 (used_bids로 화구당 1개, used_weights로 검출당
            # 1개 화구에만 배정 — 2개의 weight-class 후보가 같은 화구 window에 동시에 잡혀도
            # 아래 그리디에서 면적 큰 쪽만 살아남고 작은 쪽은 자동으로 버려짐).
            candidates.sort(key=lambda t: t[0], reverse=True)  # 면적 큰 후보부터 배정
            used_weights: set[int] = set()
            used_bids:    set[int] = set()
            for _, bid, wi in candidates:
                if bid in used_bids or wi in used_weights:
                    continue
                w = weight_like[wi]
                if w.class_id == CLASS_VENT and bid in self.last_vent_boxes:
                    # vent가 딸랑이로 승격된 경우 — 같은 박스를 vent 오버레이에도 중복 표시하지 않음
                    promoted = (int(w.x1), int(w.y1), int(w.x2), int(w.y2))
                    remaining = [b for b in self.last_vent_boxes[bid] if b != promoted]
                    if remaining:
                        self.last_vent_boxes[bid] = remaining
                    else:
                        self.last_vent_boxes.pop(bid, None)
                    self._registry.get(bid).vent_count = len(remaining)
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

            # 완료 상태 화구: weight ROI 직접 탐지 (body 종속 없이 독립 체크)
            # WAIT_SECOND는 body 기반으로 pot_present 판단 (weight ROI 이탈로 인한 오전환 방지)
            _DONE_STATES = (BurnerState.DONE_FIRST, BurnerState.DONE_SECOND)
            weight_in_roi: dict[int, bool] = {}
            for bid in burner_ids:
                if self._registry.get(bid).state not in _DONE_STATES:
                    continue
                roi = self._burner_map[bid].get("roi")
                if not roi:
                    continue
                rx, ry, rw, rh = roi
                weight_in_roi[bid] = any(
                    rx <= w.cx <= rx + rw and ry <= w.cy <= ry + rh
                    for w in weight_like
                )

            # ── Phase 2: optical flow 움직임 판별 ────────────────────────
            for bid in burner_ids:
                if bid in matched_bodies and bid in matched_has_weight:
                    x1, y1, x2, y2 = matched_bodies[bid]
                    has_wt, w_box, mask_xy = matched_has_weight[bid]
                    self.last_matched_boxes[bid] = (x1, y1, x2, y2)
                    self._body_ttl[bid] = 5

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

                    if bid in weight_in_roi:
                        detections[bid] = (weight_in_roi[bid], vibrating)
                    else:
                        detections[bid] = (True, vibrating)

                    bsm = self._registry.get(bid)
                    bsm.weight_detected = has_wt
                    bsm.vibration_score = self._oflow[bid].score
                    # current_angle 에 smoothed RMS 저장 (UI 표시용)
                    bsm.current_angle   = self._oflow[bid].last_smoothed_rms if has_wt else None
                    bsm.angle_deviation = self._oflow[bid].last_normalized_rms
                    bsm.raw_rms         = self._oflow[bid].last_rms
                    bsm.mask_px         = self._oflow[bid].last_mask_px

                else:
                    if bid in weight_in_roi:
                        # 완료 상태: body 없어도 weight ROI 직접 탐지로 pot_present 결정
                        vibrating, _ = self._oflow[bid].update(stabilized, None)
                        detections[bid] = (weight_in_roi[bid], vibrating)
                        bsm = self._registry.get(bid)
                        bsm.weight_detected = weight_in_roi[bid]
                        bsm.vibration_score = self._oflow[bid].score
                    elif self._body_ttl.get(bid, 0) > 0 and bid in self.last_matched_boxes:
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
                        bsm.vent_count = 0
                        bsm.weight_class_count = 0
                        self.last_matched_boxes.pop(bid, None)
                        self.last_weight_boxes.pop(bid, None)
                        self.last_mask_xys.pop(bid, None)
                        self.last_vent_boxes.pop(bid, None)
                        self.last_weight_candidate_boxes.pop(bid, None)

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
