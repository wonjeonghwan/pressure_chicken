from __future__ import annotations

from dataclasses import dataclass
import math

from core.detector import CLASS_POT_BODY, CLASS_POT_WEIGHT, Detection


Box = tuple[int, int, int, int]


@dataclass
class TargetSelection:
    weight_box: Box | None
    body_box: Box | None
    candidate_count: int
    jump_px: float
    reason: str


class FlowTargetSelector:
    """
    Select one stable pot-weight target for optical-flow testing.

    Strategies:
      - highest: highest confidence weight on every frame
      - nearest: nearest weight to the previous selection
      - gated: nearest weight, but reject far retargets
      - body: prefer the weight attached to the same pot/body and keep continuity

    If burner_cfg is provided, body selection is anchored to that burner ROI first.
    """

    STRATEGIES = ("highest", "nearest", "gated", "body")

    def __init__(
        self,
        strategy: str = "body",
        burner_cfg: dict | None = None,
        track_gate_px: float = 120.0,
        roi_margin_ratio: float = 0.20,
        body_expand_ratio: float = 0.15,
        body_ttl_frames: int = 15,
    ) -> None:
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unsupported strategy: {strategy}")

        self._strategy = strategy
        self._track_gate_px = float(track_gate_px)
        self._roi_margin_ratio = float(roi_margin_ratio)
        self._body_expand_ratio = float(body_expand_ratio)
        self._body_ttl_frames = int(body_ttl_frames)

        roi = burner_cfg.get("roi") if burner_cfg else None
        self._roi: tuple[int, int, int, int] | None = tuple(roi) if roi else None

        self._prev_weight_box: Box | None = None
        self._prev_body_box: Box | None = None
        self._body_ttl = 0

        self.last_selection: TargetSelection = TargetSelection(
            weight_box=None,
            body_box=None,
            candidate_count=0,
            jump_px=0.0,
            reason="init",
        )

    def reset(self) -> None:
        self._prev_weight_box = None
        self._prev_body_box = None
        self._body_ttl = 0
        self.last_selection = TargetSelection(
            weight_box=None,
            body_box=None,
            candidate_count=0,
            jump_px=0.0,
            reason="reset",
        )

    @property
    def roi_box(self) -> Box | None:
        if self._roi is None:
            return None
        x, y, w, h = self._roi
        return (x, y, x + w, y + h)

    def select(self, detections: list[Detection]) -> TargetSelection:
        bodies = [d for d in detections if d.class_id == CLASS_POT_BODY and d.confidence >= 0.30]
        weights = [d for d in detections if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]

        if self._strategy == "highest":
            return self._select_simple(weights, use_gate=False, use_prev=False, reason="highest_conf")
        if self._strategy == "nearest":
            return self._select_simple(weights, use_gate=False, use_prev=True, reason="nearest_prev")
        if self._strategy == "gated":
            return self._select_simple(weights, use_gate=True, use_prev=True, reason="gated_prev")
        return self._select_body_guided(bodies, weights)

    def _select_simple(
        self,
        weights: list[Detection],
        *,
        use_gate: bool,
        use_prev: bool,
        reason: str,
    ) -> TargetSelection:
        if not weights:
            return self._keep_missing("no_weight")

        candidates = self._weights_in_roi(weights)
        if not candidates:
            return self._keep_missing("no_weight_in_roi")

        if not use_prev or self._prev_weight_box is None:
            chosen = max(candidates, key=lambda d: d.confidence)
            return self._commit(chosen, None, len(candidates), f"{reason}:init")

        prev_cx, prev_cy = self._box_center(self._prev_weight_box)
        chosen = min(candidates, key=lambda d: self._distance_to_point(d, prev_cx, prev_cy))
        jump_px = self._distance_to_point(chosen, prev_cx, prev_cy)
        if use_gate and jump_px > self._track_gate_px:
            return self._keep_missing(f"{reason}:gated_out")

        return self._commit(chosen, None, len(candidates), reason)

    def _select_body_guided(
        self,
        bodies: list[Detection],
        weights: list[Detection],
    ) -> TargetSelection:
        if not weights:
            return self._keep_missing("body:no_weight")

        roi_weights = self._weights_in_roi(weights)
        if self._roi is not None:
            candidate_weights = roi_weights
        else:
            candidate_weights = roi_weights if roi_weights else list(weights)
        body = self._select_body(bodies, candidate_weights)

        if body is not None:
            body_weights = [w for w in candidate_weights if self._inside_body(body, w)]
            if body_weights:
                candidate_weights = body_weights

        if not candidate_weights:
            return self._keep_missing("body:no_candidate")

        if self._prev_weight_box is None:
            if body is not None:
                body_cx, body_cy = self._box_center(body)
                chosen = min(
                    candidate_weights,
                    key=lambda d: (
                        self._distance_to_point(d, body_cx, body_cy),
                        -d.confidence,
                    ),
                )
                return self._commit(chosen, body, len(candidate_weights), "body:init_from_body")

            chosen = max(candidate_weights, key=lambda d: d.confidence)
            return self._commit(chosen, body, len(candidate_weights), "body:init_fallback")

        prev_cx, prev_cy = self._box_center(self._prev_weight_box)
        chosen = min(candidate_weights, key=lambda d: self._distance_to_point(d, prev_cx, prev_cy))
        jump_px = self._distance_to_point(chosen, prev_cx, prev_cy)
        if jump_px > self._track_gate_px:
            return self._keep_missing("body:gated_out")

        return self._commit(chosen, body, len(candidate_weights), "body:tracked")

    def _select_body(
        self,
        bodies: list[Detection],
        weights: list[Detection],
    ) -> Box | None:
        if not bodies:
            return self._reuse_prev_body()

        roi_bodies = self._bodies_in_roi(bodies)
        if self._roi is not None:
            candidates = roi_bodies
        else:
            candidates = roi_bodies if roi_bodies else bodies

        if not candidates:
            return self._reuse_prev_body()

        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            rcx = rx + rw / 2.0
            rcy = ry + rh / 2.0
            chosen = min(candidates, key=lambda d: self._distance_to_point(d, rcx, rcy))
            box = self._det_box(chosen)
            self._prev_body_box = box
            self._body_ttl = self._body_ttl_frames
            return box

        if self._prev_body_box is not None:
            pcx, pcy = self._box_center(self._prev_body_box)
            chosen = min(candidates, key=lambda d: self._distance_to_point(d, pcx, pcy))
            jump_px = self._distance_to_point(chosen, pcx, pcy)
            if jump_px <= self._track_gate_px * 1.5:
                box = self._det_box(chosen)
                self._prev_body_box = box
                self._body_ttl = self._body_ttl_frames
                return box

        if self._prev_weight_box is not None:
            pcx, pcy = self._box_center(self._prev_weight_box)
            chosen = min(candidates, key=lambda d: self._distance_to_point(d, pcx, pcy))
            box = self._det_box(chosen)
            self._prev_body_box = box
            self._body_ttl = self._body_ttl_frames
            return box

        if weights:
            seed = max(weights, key=lambda d: d.confidence)
            scx, scy = self._det_center(seed)
            chosen = min(candidates, key=lambda d: self._distance_to_point(d, scx, scy))
            box = self._det_box(chosen)
            self._prev_body_box = box
            self._body_ttl = self._body_ttl_frames
            return box

        return self._reuse_prev_body()

    def _reuse_prev_body(self) -> Box | None:
        if self._prev_body_box is None or self._body_ttl <= 0:
            return None
        self._body_ttl -= 1
        return self._prev_body_box

    def _weights_in_roi(self, weights: list[Detection]) -> list[Detection]:
        if self._roi is None:
            return list(weights)

        rx, ry, rw, rh = self._roi
        mx = rw * self._roi_margin_ratio
        my = rh * self._roi_margin_ratio
        out = []
        for w in weights:
            cx, cy = self._det_center(w)
            if rx - mx <= cx <= rx + rw + mx and ry - my <= cy <= ry + rh + my:
                out.append(w)
        return out

    def _bodies_in_roi(self, bodies: list[Detection]) -> list[Detection]:
        if self._roi is None:
            return list(bodies)

        rx, ry, rw, rh = self._roi
        mx = rw * self._roi_margin_ratio
        my = rh * self._roi_margin_ratio
        out = []
        for b in bodies:
            cx, cy = self._det_center(b)
            if rx - mx <= cx <= rx + rw + mx and ry - my <= cy <= ry + rh + my:
                out.append(b)
        return out

    def _inside_body(self, body_box: Box, weight: Detection) -> bool:
        x1, y1, x2, y2 = body_box
        ex = (x2 - x1) * self._body_expand_ratio
        ey = (y2 - y1) * self._body_expand_ratio
        cx, cy = self._det_center(weight)
        return x1 - ex <= cx <= x2 + ex and y1 - ey <= cy <= y2 + ey

    def _commit(
        self,
        weight: Detection,
        body_box: Box | None,
        candidate_count: int,
        reason: str,
    ) -> TargetSelection:
        weight_box = self._det_box(weight)
        jump_px = 0.0
        if self._prev_weight_box is not None:
            jump_px = self._box_distance(weight_box, self._prev_weight_box)

        self._prev_weight_box = weight_box
        if body_box is not None:
            self._prev_body_box = body_box
            self._body_ttl = self._body_ttl_frames

        self.last_selection = TargetSelection(
            weight_box=weight_box,
            body_box=body_box,
            candidate_count=candidate_count,
            jump_px=jump_px,
            reason=reason,
        )
        return self.last_selection

    def _keep_missing(self, reason: str) -> TargetSelection:
        self.last_selection = TargetSelection(
            weight_box=None,
            body_box=self._prev_body_box,
            candidate_count=0,
            jump_px=0.0,
            reason=reason,
        )
        return self.last_selection

    @staticmethod
    def _det_box(det: Detection) -> Box:
        return (int(det.x1), int(det.y1), int(det.x2), int(det.y2))

    @staticmethod
    def _det_center(det: Detection) -> tuple[float, float]:
        return ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)

    @staticmethod
    def _box_center(box: Box) -> tuple[float, float]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @classmethod
    def _box_distance(cls, a: Box, b: Box) -> float:
        ax, ay = cls._box_center(a)
        bx, by = cls._box_center(b)
        return math.hypot(ax - bx, ay - by)

    @classmethod
    def _distance_to_point(cls, det: Detection, px: float, py: float) -> float:
        cx, cy = cls._det_center(det)
        return math.hypot(cx - px, cy - py)
