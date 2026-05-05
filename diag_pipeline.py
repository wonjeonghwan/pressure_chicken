"""
파이프라인 시각화 — 입력부터 출력까지 각 중간값을 4개 그래프로 출력

  Row 1  raw RMS vs EMA-smoothed RMS vs threshold        ← Step 5 (rms_ema_alpha)
  Row 2  window 내 motion 비율 vs trigger 기준선          ← 최종 STEAMING 판정
  Row 3  bbox center raw_cx vs EMA_cx                    ← Step 2 (pos_ema_alpha)
  Row 4  딸랑이 감지 여부 & STEAMING 최종 출력

사용:
    uv run python diag_pipeline.py --burner 1 --frames 300
    uv run python diag_pipeline.py --burner 1 --frames 300 --skip 200 --save out.png
"""

import argparse
import json
import math
import sys

import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Windows 한글 폰트 설정
_font_path = r"C:\Windows\Fonts\malgun.ttf"
try:
    fm.fontManager.addfont(_font_path)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_font_path).get_name()
except Exception:
    pass  # 폰트 없으면 경고만 나오고 동작은 함

sys.path.insert(0, ".")

from core.detector import CLASS_POT_BODY, CLASS_POT_WEIGHT, BurnerDetector
from core.optical_flow import OpticalFlowDetector
from core.stabilizer import Stabilizer
from sources.video_source import VideoSource


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burner", type=int, required=True, help="분석할 화구 번호")
    parser.add_argument("--frames", type=int, default=300, help="분석할 프레임 수")
    parser.add_argument("--skip",   type=int, default=0,   help="건너뛸 초기 프레임 수")
    parser.add_argument("--config", default="config/store_config.json")
    parser.add_argument("--save",   default=None, help="저장할 png 경로 (없으면 화면 출력)")
    args = parser.parse_args()

    config   = load_config(args.config)
    bid      = args.burner
    bcfg     = next((b for b in config["burners"] if b["id"] == bid), None)
    if bcfg is None:
        print(f"화구 {bid} 없음"); return

    flow_cfg = config.get("optical_flow", {})
    rms_thr  = flow_cfg.get("rms_threshold",  0.6)
    trigger  = flow_cfg.get("trigger_frames", 14)
    window   = flow_cfg.get("window_frames",  25)
    rms_alpha = flow_cfg.get("rms_ema_alpha", 0.35)
    pos_alpha = flow_cfg.get("pos_ema_alpha", 0.3)

    src      = VideoSource(config["sources"][0])
    src.open()
    stab     = Stabilizer(config.get("stabilizer", {}))
    detector = BurnerDetector(config["model"]["weights"], config["model"]["confidence"])
    oflow    = OpticalFlowDetector(flow_cfg)

    roi = bcfg.get("roi")
    if not roi:
        print("ROI 없음 — calibration.py로 먼저 ROI를 설정하세요"); return
    rx_roi, ry_roi, rw_roi, rh_roi = roi

    for _ in range(args.skip):
        ret, _ = src.read()
        if not ret:
            break

    # ── 기록 배열 ──────────────────────────────────────────────────────────
    xs:          list[int]   = []
    raw_rms:     list[float] = []
    ema_rms:     list[float] = []
    window_ratio:list[float] = []
    raw_cx_list: list[float | None] = []
    ema_cx_list: list[float | None] = []
    per_motion:  list[bool]  = []   # ema_rms > threshold (프레임 단위)
    steaming:    list[bool]  = []   # window 투표 최종 판정
    has_weight:  list[bool]  = []

    # ── 프레임 루프 ────────────────────────────────────────────────────────
    frame_n = 0
    while frame_n < args.frames:
        ret, frame = src.read()
        if not ret or frame is None:
            break
        frame_n += 1
        abs_frame = frame_n + args.skip

        stabilized = stab.stabilize(frame)
        fh, fw = stabilized.shape[:2]
        margin = 50

        cx1 = max(0, rx_roi - margin)
        cy1 = max(0, ry_roi - margin)
        cx2 = min(fw, rx_roi + rw_roi + margin)
        cy2 = min(fh, ry_roi + rh_roi + margin)
        crop = stabilized[cy1:cy2, cx1:cx2]

        dets = detector.detect(crop)
        for d in dets:
            d.x1 += cx1; d.x2 += cx1
            d.y1 += cy1; d.y2 += cy1
            if d.mask_xy is not None:
                d.mask_xy = d.mask_xy + np.array([[cx1, cy1]], dtype=np.float32)

        bodies  = [d for d in dets if d.class_id == CLASS_POT_BODY   and d.confidence >= 0.3]
        weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]

        # 밥솥 매칭
        mx, my    = rw_roi * 0.2, rh_roi * 0.2
        anchor    = (rx_roi + rw_roi / 2, ry_roi + rh_roi / 2)
        best_body = None
        best_dist = float("inf")
        for b in bodies:
            if (rx_roi - mx <= b.cx <= rx_roi + rw_roi + mx and
                    ry_roi - my <= b.cy <= ry_roi + rh_roi + my):
                dist = math.hypot(b.cx - anchor[0], b.cy - anchor[1])
                if dist < best_dist:
                    best_dist, best_body = dist, b

        def _record(raw_cx_val: float | None, is_weight: bool, vibrating: bool) -> None:
            xs.append(abs_frame)
            raw_rms.append(oflow.last_rms)
            ema_rms.append(oflow.last_smoothed_rms)
            votes, _, _ = oflow.window_votes
            window_ratio.append(votes / window)
            raw_cx_list.append(raw_cx_val)
            ema_cx_list.append(oflow._ema_cx)
            per_motion.append(bool(oflow.last_smoothed_rms > rms_thr))
            steaming.append(vibrating)
            has_weight.append(is_weight)

        if best_body is None:
            vib, _ = oflow.update(stabilized, None)
            _record(None, False, vib)
            continue

        # 딸랑이 매칭
        bx1, by1, bx2, by2 = best_body.x1, best_body.y1, best_body.x2, best_body.y2
        bcx = (bx1 + bx2) / 2
        we = (bx2 - bx1) * 0.15
        he = (by2 - by1) * 0.15
        best_weight = None
        best_wd = float("inf")
        for w in weights:
            if bx1 - we <= w.cx <= bx2 + we and by1 - he <= w.cy <= by2 + he:
                d = abs(w.cx - bcx)
                if d < best_wd:
                    best_wd, best_weight = d, w

        if best_weight is None:
            vib, _ = oflow.update(stabilized, None)
            _record(None, False, vib)
            continue

        wx1, wy1, wx2, wy2 = best_weight.x1, best_weight.y1, best_weight.x2, best_weight.y2
        raw_cx_val = (wx1 + wx2) / 2.0

        vib, _ = oflow.update(
            stabilized,
            (int(wx1), int(wy1), int(wx2), int(wy2)),
            best_weight.mask_xy,
        )
        _record(raw_cx_val, True, vib)

    src.release()

    if not xs:
        print("기록된 프레임 없음"); return

    # ── 시각화 ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), sharex=True)
    fig.suptitle(
        f"파이프라인 시각화 — 화구 {bid}  |  "
        f"rms_α={rms_alpha}  pos_α={pos_alpha}  thr={rms_thr}  "
        f"trigger={trigger}/{window}",
        fontsize=13,
    )

    def _shade_steaming(ax: plt.Axes) -> None:
        in_s = False
        for i, s in enumerate(steaming):
            if s and not in_s:
                s_start = xs[i]; in_s = True
            elif not s and in_s:
                ax.axvspan(s_start, xs[i - 1], alpha=0.12, color="red")
                in_s = False
        if in_s:
            ax.axvspan(s_start, xs[-1], alpha=0.12, color="red")

    # Row 1 ── raw RMS / EMA RMS / threshold ─────────────────────────────
    ax = axes[0]
    ax.plot(xs, raw_rms, color="steelblue",  alpha=0.55, linewidth=0.9,
            label=f"raw RMS  x_t")
    ax.plot(xs, ema_rms, color="darkorange", linewidth=1.8,
            label=f"EMA RMS  S_t = {rms_alpha}·x_t + {1-rms_alpha}·S_{{t-1}}")
    ax.axhline(rms_thr, color="red", linestyle="--", linewidth=1.2,
               label=f"threshold = {rms_thr}")
    _shade_steaming(ax)
    ax.set_ylabel("RMS (픽셀/프레임)")
    ax.set_title("Step 5 — raw RMS → EMA 스무딩 → threshold 비교 (프레임 단위 motion 판정)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 2 ── window 투표 ────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(xs, window_ratio, color="forestgreen", linewidth=1.5,
            label=f"window 내 motion 비율  (분자/분모 = True수/{window})")
    ax.axhline(trigger / window, color="red", linestyle="--", linewidth=1.2,
               label=f"trigger 기준  {trigger}/{window} = {trigger/window:.2f}")
    _shade_steaming(ax)
    ax.fill_between(xs, window_ratio,
                    [trigger / window] * len(xs),
                    where=[w >= trigger / window for w in window_ratio],
                    alpha=0.25, color="red", label="STEAMING 구간")
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel("비율")
    ax.set_title(f"최종 판정 — window {window}프레임 중 {trigger}개 이상 motion=True → STEAMING")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 3 ── bbox center EMA (Step 2) ───────────────────────────────────
    ax = axes[2]
    raw_cx_np = np.array([v if v is not None else np.nan for v in raw_cx_list], dtype=float)
    ema_cx_np = np.array([v if v is not None else np.nan for v in ema_cx_list], dtype=float)
    ax.plot(xs, raw_cx_np, color="steelblue",  alpha=0.5, linewidth=0.9,
            label=f"raw bbox cx  (YOLO 직접 출력)")
    ax.plot(xs, ema_cx_np, color="darkorange", linewidth=1.8,
            label=f"EMA cx  = {pos_alpha}·raw_cx + {1-pos_alpha}·ema_cx_prev  (crop 중심)")
    # 잡음 폭 시각화: raw - ema 차이
    diff = np.abs(raw_cx_np - ema_cx_np)
    ax2 = ax.twinx()
    ax2.fill_between(xs, diff, alpha=0.15, color="purple",
                     label=f"|raw−EMA| (px)")
    ax2.set_ylabel("|raw − EMA| (px)", color="purple", fontsize=8)
    ax2.tick_params(axis="y", labelcolor="purple", labelsize=7)
    ax.set_ylabel("x 좌표 (픽셀)")
    ax.set_title(f"Step 2 — YOLO bbox center EMA (pos_α={pos_alpha})  →  crop 위치 안정화")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 4 ── 최종 상태 ──────────────────────────────────────────────────
    ax = axes[3]
    has_wt_int = [1 if v else 0 for v in has_weight]
    steam_int  = [2 if v else 0 for v in steaming]
    ax.fill_between(xs, has_wt_int, step="mid", alpha=0.45, color="royalblue",
                    label="딸랑이 감지됨 (YOLO weight 매칭)")
    ax.fill_between(xs, steam_int,  step="mid", alpha=0.55, color="crimson",
                    label="STEAMING 판정 (window 투표 통과)")
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["없음", "딸랑이\n감지", "STEAMING"])
    ax.set_xlabel("프레임 번호")
    ax.set_title("최종 출력 — 딸랑이 YOLO 감지 & STEAMING 확정")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"저장됨: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
