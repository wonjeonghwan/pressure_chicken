from __future__ import annotations

import argparse
import json
import os
import sys

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
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--strategy", default="body", choices=FlowTargetSelector.STRATEGIES)
    parser.add_argument("--burner-id", type=int, default=None)
    parser.add_argument("--track-gate-px", type=float, default=120.0)
    parser.add_argument("--compare-strategies", action="store_true")
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


def percentile_or_zero(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), q))


def mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def run_bench(
    cfg: dict,
    video_path: str,
    frames: int,
    strategy: str,
    burner_cfg: dict | None,
    track_gate_px: float,
    verbose: bool,
) -> dict:
    model_cfg = cfg.get("model", {})
    stab = Stabilizer(cfg.get("stabilizer", {}))
    det = BurnerDetector(
        model_cfg.get("weights", "models/pot_seg.pt"),
        model_cfg.get("confidence", 0.5),
    )
    selector = FlowTargetSelector(
        strategy=strategy,
        burner_cfg=burner_cfg,
        track_gate_px=track_gate_px,
    )
    oflow = OpticalFlowDetector(cfg.get("optical_flow", {}))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if verbose:
        print(f"\n[Video] {video_path}  {vid_w}x{vid_h}  {fps:.1f}fps")
        print(
            f"[Target] strategy={strategy}  burner_id={burner_cfg.get('id') if burner_cfg else '-'}  "
            f"track_gate={track_gate_px:.0f}px"
        )
        print(
            f"[Flow] rms_threshold={cfg['optical_flow']['rms_threshold']}  "
            f"window={cfg['optical_flow']['window_frames']}  "
            f"trigger={cfg['optical_flow']['trigger_frames']}\n"
        )
        print(
            f"{'frame':>6}  {'w_box':>22}  {'sel_jump':>8}  {'rms':>7}  "
            f"{'skip':>5}  {'motion':>7}  {'score':>7}  {'reason':>18}"
        )
        print("-" * 110)

    frame_idx = 0
    rms_log: list[float] = []
    selected_frames = 0
    missing_frames = 0
    selector_jumps_gt100 = 0
    selector_jump_log: list[float] = []
    oflow_skip_frames = 0
    motion_true_frames = 0

    while frame_idx < frames:
        ret, frame = cap.read()
        if not ret:
            break

        stabilized = stab.stabilize(frame)
        detections = det.detect(stabilized)
        selection = selector.select(detections)
        motion, rms = oflow.update(stabilized, selection.weight_box)

        if selection.weight_box is not None:
            selected_frames += 1
            selector_jump_log.append(selection.jump_px)
            if selection.jump_px > 100.0:
                selector_jumps_gt100 += 1
        else:
            missing_frames += 1

        if oflow.last_skipped:
            oflow_skip_frames += 1
        if motion:
            motion_true_frames += 1
        if rms > 0:
            rms_log.append(rms)

        if verbose and frame_idx % 15 == 0:
            box = selection.weight_box
            box_str = f"({box[0]},{box[1]},{box[2]},{box[3]})" if box else "  not selected "
            skip_str = "SKIP" if oflow.last_skipped else ""
            print(
                f"{frame_idx:>6}  {box_str:>22}  {selection.jump_px:>8.1f}  {rms:>7.3f}  "
                f"{skip_str:>5}  {'True' if motion else 'False':>7}  {oflow.score:>7.2f}  "
                f"{selection.reason:>18}"
            )

        frame_idx += 1

    cap.release()

    return {
        "frames": frame_idx,
        "selected_frames": selected_frames,
        "missing_frames": missing_frames,
        "selector_jumps_gt100": selector_jumps_gt100,
        "selector_jump_mean": mean_or_zero(selector_jump_log),
        "oflow_skip_frames": oflow_skip_frames,
        "motion_true_frames": motion_true_frames,
        "final_score": oflow.score,
        "rms_mean": mean_or_zero(rms_log),
        "rms_std": float(np.std(np.array(rms_log))) if rms_log else 0.0,
        "rms_min": float(np.min(np.array(rms_log))) if rms_log else 0.0,
        "rms_max": float(np.max(np.array(rms_log))) if rms_log else 0.0,
        "rms_p60": percentile_or_zero(rms_log, 60),
        "rms_p95": percentile_or_zero(rms_log, 95),
    }


def print_summary(strategy: str, summary: dict) -> None:
    print("\n" + "=" * 72)
    print(f"[Summary] strategy={strategy}  frames={summary['frames']}\n")
    print(
        f"  target selected     : {summary['selected_frames']} "
        f"({summary['selected_frames'] / max(summary['frames'], 1):.1%})"
    )
    print(
        f"  target missing      : {summary['missing_frames']} "
        f"({summary['missing_frames'] / max(summary['frames'], 1):.1%})"
    )
    print(f"  selector jump >100  : {summary['selector_jumps_gt100']}")
    print(f"  selector jump mean  : {summary['selector_jump_mean']:.2f}px")
    print(f"  oflow skipped       : {summary['oflow_skip_frames']}")
    print(f"  motion true frames  : {summary['motion_true_frames']}")
    print(f"  final score         : {summary['final_score']:.2f}")
    print()
    print(f"  RMS mean            : {summary['rms_mean']:.4f}")
    print(f"  RMS std             : {summary['rms_std']:.4f}")
    print(f"  RMS min             : {summary['rms_min']:.4f}")
    print(f"  RMS max             : {summary['rms_max']:.4f}")
    print(f"  RMS p60             : {summary['rms_p60']:.4f}")
    print(f"  RMS p95             : {summary['rms_p95']:.4f}")
    print()
    print(f"  [Suggested rms_threshold] ~{summary['rms_p60']:.2f}")
    print("=" * 72)


def print_compare_table(results: dict[str, dict]) -> None:
    print("\n[Strategy Compare]")
    print(
        f"{'strategy':<12} {'select%':>8} {'miss%':>8} {'jump>100':>10} "
        f"{'of_skip%':>9} {'rms_mean':>9} {'motion':>8} {'score':>7}"
    )
    print("-" * 78)
    for strategy, summary in results.items():
        frames = max(summary["frames"], 1)
        print(
            f"{strategy:<12} "
            f"{summary['selected_frames'] / frames:>8.1%} "
            f"{summary['missing_frames'] / frames:>8.1%} "
            f"{summary['selector_jumps_gt100']:>10} "
            f"{summary['oflow_skip_frames'] / frames:>9.1%} "
            f"{summary['rms_mean']:>9.3f} "
            f"{summary['motion_true_frames']:>8} "
            f"{summary['final_score']:>7.2f}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_cfg()
    burner_cfg = find_burner_cfg(cfg, args.burner_id)

    if args.compare_strategies:
        results: dict[str, dict] = {}
        for strategy in FlowTargetSelector.STRATEGIES:
            results[strategy] = run_bench(
                cfg=cfg,
                video_path=args.video,
                frames=args.frames,
                strategy=strategy,
                burner_cfg=burner_cfg,
                track_gate_px=args.track_gate_px,
                verbose=False,
            )
        print_compare_table(results)
        for strategy, summary in results.items():
            print_summary(strategy, summary)
        return

    summary = run_bench(
        cfg=cfg,
        video_path=args.video,
        frames=args.frames,
        strategy=args.strategy,
        burner_cfg=burner_cfg,
        track_gate_px=args.track_gate_px,
        verbose=True,
    )
    print_summary(args.strategy, summary)


if __name__ == "__main__":
    main()
