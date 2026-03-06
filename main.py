"""
압력밥솥 타이머 시스템 — 진입점

실행:
    uv run python main.py                                  # 기본 config
    uv run python main.py --calibrate                      # ROI 캘리브레이션
    uv run python main.py --config config/store_001.json   # 매장 config 지정
    uv run python main.py --source-0 video_a.mp4           # 영상 파일 지정
    uv run python main.py --test 60                        # 60프레임 후 자동 종료 (테스트용)
"""

import argparse
import json
import os
import sys

# Windows 터미널 한국어 출력 지원
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pygame

from sources.video_source import VideoSource
from core.state_machine import BurnerRegistry
from core.detector import BurnerDetector
from core.frame_processor import FrameProcessor
from ui.ui_display import UIDisplay

DEFAULT_CONFIG = "config/store_config.json"


# ── config ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[main] config 없음: {path}")
        print("       먼저 'python main.py --calibrate' 를 실행하거나")
        print("       config/store_config.json 을 확인하세요.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_source_overrides(config: dict, overrides: dict[int, str]) -> None:
    for src in config["sources"]:
        if src["id"] in overrides:
            src["type"] = "file"
            src["path"] = overrides[src["id"]]


# ── 메인 루프 ─────────────────────────────────────────────────────────────────

def run(config: dict, test_frames: int = 0) -> None:
    sources_cfg  = config["sources"]
    burners_cfg  = config["burners"]
    motion_cfg   = config.get("motion", {})
    ui_cfg       = config.get("ui", {})
    model_cfg    = config.get("model", {})

    weights    = model_cfg.get("weights",    "models/pot_detector.pt")
    confidence = model_cfg.get("confidence", 0.5)

    # 1) VideoSource
    sources: dict[int, VideoSource] = {}
    for sc in sources_cfg:
        vs = VideoSource(sc)
        vs.open()
        sources[sc["id"]] = vs

    # 2) BurnerRegistry
    registry    = BurnerRegistry()
    burner_meta: dict[int, dict] = {}
    for b in burners_cfg:
        registry.add(
            b["id"],
            b.get("countdown_first",  b.get("countdown_seconds", 720)),
            b.get("countdown_second", 300),
        )
        burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}

    # 3) Detector (YOLO or OpenCV fallback)
    detector = BurnerDetector(weights, confidence, motion_cfg)

    # 4) FrameProcessor
    processor = FrameProcessor(sources, burners_cfg, registry, detector, motion_cfg)

    # 5) UI
    display = UIDisplay(ui_cfg, registry, burner_meta, model_missing=detector.model_missing)
    display.init()

    print("[main] 시작. ESC 로 종료.")
    if detector.model_missing:
        print("[main] 테스트 모드: 자동 감지 비활성화. ▶ 버튼으로 수동 시작 가능.")

    frame_count = 0
    running     = True
    while running:
        for event in pygame.event.get():
            if display.handle_event(event):
                running = False

        processor.step()
        display.render()

        frame_count += 1
        if test_frames > 0 and frame_count >= test_frames:
            print(f"[main] 테스트 {test_frames}프레임 완료. 정상 종료.")
            running = False

    display.quit()
    for vs in sources.values():
        vs.release()

    print("[main] 종료.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="압력밥솥 타이머 시스템")
    parser.add_argument("--config",    default=DEFAULT_CONFIG)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--source-0",  dest="source_0", default=None)
    parser.add_argument("--source-1",  dest="source_1", default=None)
    parser.add_argument("--test",      type=int, default=0, metavar="N",
                        help="N 프레임 처리 후 자동 종료 (0=무한)")
    args = parser.parse_args()

    if args.calibrate:
        from calibration import run_calibration
        run_calibration(args.config)
        return

    config = load_config(args.config)

    overrides: dict[int, str] = {}
    if args.source_0:
        overrides[0] = args.source_0
    if args.source_1:
        overrides[1] = args.source_1
    if overrides:
        apply_source_overrides(config, overrides)

    run(config, test_frames=args.test)


if __name__ == "__main__":
    main()
