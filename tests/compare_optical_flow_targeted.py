from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.detector import BurnerDetector
from core.flow_target import FlowTargetSelector
from core.optical_flow import OpticalFlowDetector
from core.stabilizer import Stabilizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="raw/Sample01.mp4")
    parser.add_argument("--display-w", type=int, default=1600)
    parser.add_argument("--graph-sec", type=float, default=10.0)
    parser.add_argument("--strategy", default="body", choices=FlowTargetSelector.STRATEGIES)
    parser.add_argument("--burner-id", type=int, default=None)
    parser.add_argument("--track-gate-px", type=float, default=120.0)
    return parser.parse_args()


def load_cfg() -> dict:
    with open("config/store_config.json", encoding="utf-8") as f:
        return json.load(f)


def find_burner_cfg(cfg: dict, burner_id: int | None) -> dict | None:
    if burner_id is None:
        return None
    for burner in cfg.get("burners", []):
        if burner.get("id") == burner_id:
            return burner
    raise ValueError(f"burner_id={burner_id} not found in config/store_config.json")


def draw_box(frame: np.ndarray, box: tuple[int, int, int, int], color: tuple[int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    burner_cfg = find_burner_cfg(cfg, args.burner_id)
    model_cfg = cfg.get("model", {})

    stab = Stabilizer(cfg.get("stabilizer", {}))
    det = BurnerDetector(
        model_cfg.get("weights", "models/pot_seg.pt"),
        model_cfg.get("confidence", 0.5),
    )
    selector = FlowTargetSelector(
        strategy=args.strategy,
        burner_cfg=burner_cfg,
        track_gate_px=args.track_gate_px,
    )
    oflow = OpticalFlowDetector(cfg.get("optical_flow", {}))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[Video] {args.video}  {vid_w}x{vid_h}  {total}f  {fps:.1f}fps  "
        f"strategy={args.strategy}  burner_id={burner_cfg.get('id') if burner_cfg else '-'}"
    )
    print("  space: pause  ->: step  q/ESC: quit")

    panel_h = 86
    graph_h = 100
    roi_w = 220
    half_w = (args.display_w - roi_w) // 2
    scale = half_w / vid_w
    disp_h = int(vid_h * scale)

    graph_len = max(1, int(fps * args.graph_sec))
    rms_hist: deque[float] = deque([0.0] * graph_len, maxlen=graph_len)
    score_hist: deque[float] = deque([0.0] * graph_len, maxlen=graph_len)

    frame_idx = 0
    paused = False
    last_frame = None
    last_stab = None
    last_selection = None
    last_motion = False
    last_rms = 0.0
    last_smoothed_rms = 0.0
    smoothed_rms_hist: deque[float] = deque([0.0] * graph_len, maxlen=graph_len)

    def process_frame(frame: np.ndarray) -> None:
        nonlocal last_frame, last_stab, last_selection, last_motion, last_rms, last_smoothed_rms
        stabilized = stab.stabilize(frame)
        detections = det.detect(stabilized)
        selection = selector.select(detections)
        motion, rms = oflow.update(stabilized, selection.weight_box)

        last_frame = frame.copy()
        last_stab = stabilized.copy()
        last_selection = selection
        last_motion = motion
        last_rms = rms
        last_smoothed_rms = oflow.last_smoothed_rms
        rms_hist.append(rms)
        smoothed_rms_hist.append(oflow.last_smoothed_rms)
        score_hist.append(oflow.score)

    def draw_panel() -> np.ndarray:
        panel = np.zeros((panel_h, args.display_w, 3), dtype=np.uint8)
        panel[:] = (25, 25, 25)

        selection = last_selection
        if selection is None:
            return panel

        motion_color = (0, 255, 60) if last_motion else (200, 200, 200)
        items = [
            ("Target", f"{args.strategy} / {selection.reason}", (220, 220, 220)),
            ("Selector", f"jump {selection.jump_px:.1f}px  cand {selection.candidate_count}", (0, 220, 255)),
            ("Raw RMS", f"{last_rms:.3f}", (100, 200, 255)),
            ("Smoothed RMS", f"{last_smoothed_rms:.3f}  score {oflow.score:.2f}  {'MOTION' if last_motion else 'still'}", motion_color),
            ("Reset", f"{oflow.last_reset_reason or '-'}  skip={oflow.last_skipped}", (255, 180, 0) if oflow.last_skipped else (160, 160, 160)),
        ]

        col_w = args.display_w // len(items)
        for i, (label, value, color) in enumerate(items):
            x = i * col_w + 10
            cv2.putText(panel, label, (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
            cv2.putText(panel, value, (x, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1)

        for i in range(1, len(items)):
            cv2.line(panel, (i * col_w, 8), (i * col_w, panel_h - 8), (60, 60, 60), 1)
        return panel

    def draw_graph() -> np.ndarray:
        graph = np.zeros((graph_h, args.display_w, 3), dtype=np.uint8)
        graph[:] = (20, 20, 20)

        max_rms = max(max(rms_hist), cfg["optical_flow"]["rms_threshold"] * 2, 1.0)
        base_y = graph_h - 15
        thr_y = int(base_y - cfg["optical_flow"]["rms_threshold"] / max_rms * (base_y - 4))
        thr_y = max(2, min(graph_h - 2, thr_y))

        cv2.line(graph, (0, base_y), (args.display_w, base_y), (70, 70, 70), 1)
        cv2.line(graph, (0, thr_y), (args.display_w, thr_y), (0, 80, 255), 1)
        cv2.putText(graph, f"thr={cfg['optical_flow']['rms_threshold']}", (4, thr_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 80, 255), 1)

        def hist_to_pts(hist: deque[float], *, invert: bool, max_val: float) -> list[tuple[int, int]]:
            pts = []
            for i, value in enumerate(hist):
                x = int(i / max(graph_len - 1, 1) * args.display_w)
                if invert:
                    y = int(base_y * (1.0 - min(value, 1.0)))
                else:
                    y = int(base_y - value / max_val * (base_y - 4))
                pts.append((x, max(2, min(graph_h - 2, y))))
            return pts

        def draw_line(pts: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
            for i in range(1, len(pts)):
                cv2.line(graph, pts[i - 1], pts[i], color, 1)

        draw_line(hist_to_pts(rms_hist, invert=False, max_val=max_rms), (80, 130, 180))       # raw RMS: dim blue
        draw_line(hist_to_pts(smoothed_rms_hist, invert=False, max_val=max_rms), (50, 220, 255))  # smoothed RMS: bright cyan
        draw_line(hist_to_pts(score_hist, invert=True, max_val=1.0), (200, 150, 255))            # score: purple

        # legend
        cv2.putText(graph, "raw", (args.display_w - 130, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 130, 180), 1)
        cv2.putText(graph, "smooth", (args.display_w - 95, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (50, 220, 255), 1)
        cv2.putText(graph, "score", (args.display_w - 48, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 150, 255), 1)
        return graph

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[End of video]")
                break
            process_frame(frame)
            frame_idx += 1

        if last_frame is None or last_stab is None or last_selection is None:
            if cv2.waitKey(10) & 0xFF in (ord("q"), 27):
                break
            continue

        vis_orig = cv2.resize(last_frame, (half_w, disp_h))
        vis_stab = cv2.resize(last_stab, (half_w, disp_h))
        roi_panel = np.zeros((disp_h, roi_w, 3), dtype=np.uint8)
        roi_panel[:] = (30, 30, 30)

        if selector.roi_box is not None:
            rx1, ry1, rx2, ry2 = selector.roi_box
            draw_box(last_frame, (rx1, ry1, rx2, ry2), (0, 180, 120), "burner roi")

        if last_selection.body_box is not None:
            draw_box(last_frame, last_selection.body_box, (255, 0, 255), "body")

        if last_selection.weight_box is not None:
            draw_box(last_frame, last_selection.weight_box, (0, 255, 255), "weight")

        vis_orig = cv2.resize(last_frame, (half_w, disp_h))
        cv2.putText(vis_orig, "Original + selector", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)
        cv2.putText(vis_orig, f"frame {frame_idx}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        cv2.putText(vis_stab, "Stabilized + flow target", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)
        motion_label = "MOTION" if last_motion else "still"
        motion_color = (0, 255, 60) if last_motion else (180, 180, 180)
        cv2.putText(
            vis_stab,
            f"raw={last_rms:.3f}  smooth={last_smoothed_rms:.3f}  score={oflow.score:.2f}  [{motion_label}]",
            (10, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            motion_color,
            1,
        )

        if last_selection.weight_box is not None and oflow.last_flow is not None and oflow.last_roi_box is not None:
            x1, y1, x2, y2 = oflow.last_roi_box
            roi_crop = last_stab[y1:y2, x1:x2]
            if roi_crop.size > 0:
                flow_bgr = OpticalFlowDetector.flow_to_bgr(oflow.last_flow)
                rh, rw = roi_crop.shape[:2]
                scale_roi = min(roi_w / max(rw, 1), (disp_h - 60) / max(rh, 1))
                nw, nh = max(1, int(rw * scale_roi)), max(1, int(rh * scale_roi))
                ox = (roi_w - nw) // 2
                oy = (disp_h - nh - 50) // 2 + 30
                blended = cv2.addWeighted(
                    cv2.resize(roi_crop, (nw, nh)),
                    0.5,
                    cv2.resize(flow_bgr, (nw, nh)),
                    0.5,
                    0,
                )
                roi_panel[oy:oy + nh, ox:ox + nw] = blended
                cv2.rectangle(roi_panel, (ox, oy), (ox + nw, oy + nh), (0, 255, 255), 1)

        cv2.putText(roi_panel, "ROI + Flow", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(roi_panel, f"reason: {last_selection.reason}", (8, disp_h - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(roi_panel, f"jump: {last_selection.jump_px:.1f}px", (8, disp_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)

        video_row = np.hstack([vis_orig, roi_panel, vis_stab])
        display = np.vstack([video_row, draw_panel(), draw_graph()])
        cv2.imshow("Phase 2 Targeted Optical Flow  [space=pause  q=quit]", display)

        key = cv2.waitKey(1 if not paused else 30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
            print(f"  {'PAUSED' if paused else 'RESUMED'}  frame={frame_idx}  rms={last_rms:.4f}")
        elif paused and key == 83:
            ret, frame = cap.read()
            if ret:
                process_frame(frame)
                frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[Done] {frame_idx} frames")


if __name__ == "__main__":
    main()
