"""
압력밥솥 타이머 시스템 — 진입점

실행:
    uv run python main.py
    uv run python main.py --config config/store_001.json
    uv run python main.py --source-0 video_a.mp4
"""

import argparse
import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import pygame

from sources.video_source import VideoSource
from sources.camera_utils import save_config, switch_camera
from core.state_machine import BurnerRegistry
from core.detector import BurnerDetector
from core.frame_processor import FrameProcessor
from ui.ui_display import UIDisplay

DEFAULT_CONFIG = "config/store_config.json"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[main] config.json 이 없습니다. 기본값으로 생성하고 시작합니다.")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        default_cfg = {
            "sources": [{"id": 0, "type": "camera", "index": 0}],
            "burners": [],
            "motion": {},
            "model": {"weights": "models/pot_detector.pt", "confidence": 0.5},
            "ui": {"window_size": [1280, 720], "grid_cols": 2}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default_cfg, f, indent=2)
        return default_cfg

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def apply_source_overrides(config: dict, overrides: dict[int, str]) -> None:
    sources = config.get("sources", [])
    for sc in sources:
        sid = sc.get("id")
        if sid in overrides:
            sc["type"] = "file"
            sc["path"] = overrides[sid]


def run(config: dict, test_frames: int = 0) -> None:
    sources_cfg  = config["sources"]
    burners_cfg  = config.get("burners", [])
    motion_cfg   = config.get("motion", {})
    ui_cfg       = config.get("ui", {})
    model_cfg    = config.get("model", {})

    weights    = model_cfg.get("weights", "models/pot_detector.pt")
    confidence = model_cfg.get("confidence", 0.5)

    _FRAME_INTERVAL = 1 / 15

    # 1) VideoSource
    sources = {}
    cam_indices = {}

    for sc in sources_cfg:
        vs = VideoSource(sc)
        vs.open()
        
        if vs.failed and sc.get("type", "camera") == "camera":
            print(f"[main] 소스 {sc['id']} 실패. 대체 카메라 탐색...")
            available = VideoSource.find_available_cameras()
            if available:
                sc["index"] = available[0]
                vs = VideoSource(sc)
                vs.open()
                
        if sc.get("type") == "file":
            target_fps = 1 / _FRAME_INTERVAL
            skip = max(1, round(vs.fps / target_fps))
            sc["_skip_frames"] = skip
            print(f"[main] 파일 소스 {sc['id']}: skip={skip}")

        sources[sc["id"]] = vs
        if sc.get("type", "camera") == "camera":
            cam_indices[sc["id"]] = sc.get("index", 0)

    # 2) Registry
    registry = BurnerRegistry()
    burner_meta = {}
    for b in burners_cfg:
        registry.add(
            b["id"],
            b.get("countdown_first", 720),
            b.get("countdown_second", 300),
        )
        burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}

    # 3) Detector
    detector = BurnerDetector(weights, confidence, motion_cfg)

    # 4) Processor
    processor = FrameProcessor(sources, burners_cfg, registry, detector, config)

    # 5) UI
    display = UIDisplay(
        ui_cfg=ui_cfg, 
        registry=registry, 
        burner_meta=burner_meta, 
        config_data=config, 
        config_path=config.get("_path"), 
        model_missing=detector.model_missing
    )
    display.init()

    print("[main] 대시보드 시작. Q 로 종료.")

    import time as _time
    _last_frame      = 0.0
    _last_detect     = 0.0
    _DETECT_INTERVAL = 1 / 15

    current_frames = {}
    frame_count = 0
    running = True
    
    _last_cam_switch = 0.0
    _CAM_SWITCH_COOLDOWN = 0.5

    def trigger_camera_switch():
        nonlocal _last_cam_switch
        now = _time.monotonic()
        if now - _last_cam_switch < _CAM_SWITCH_COOLDOWN:
            return
        _last_cam_switch = now
        for src_id in list(cam_indices):
            cam_indices[src_id] = switch_camera(
                sources, src_id, cam_indices[src_id],
                config=config, config_path=config.get("_path")
            )
            
    display.on_camera_switch = trigger_camera_switch

    def on_calib_saved(new_burners):
        nonlocal processor
        config["burners"] = new_burners
        registry.burners.clear()
        burner_meta.clear()
        for b in new_burners:
            registry.add(
                b["id"],
                b.get("countdown_first", 720),
                b.get("countdown_second", 300),
            )
            burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}
        # Re-init processor with new burners
        processor = FrameProcessor(sources, new_burners, registry, detector, config)

    display.on_calibration_saved = on_calib_saved

    try:
        while running:
            now = _time.monotonic()
            
            for event in pygame.event.get():
                if display.handle_event(event):
                    running = False

            # 만약 캘리브레이션 모드에서 값이 재설정되었다면 (심리스 부분 재실행은 추후 고도화, 현재는 UI만 유지)
            # 여기서는 비디오 일시정지 상태가 아니면 프레임 Update
            if not display.video_paused:
                if now - _last_frame >= _FRAME_INTERVAL:
                    current_frames = processor.read_frames()
                    _last_frame = now
                    frame_count += 1

                if not display.calibration_mode:
                    if now - _last_detect >= _DETECT_INTERVAL:
                        processor.detect_and_update()
                        _last_detect = now

            display.render(frames=current_frames, processor=processor)

            if test_frames > 0 and frame_count >= test_frames:
                print(f"[main] 테스트 {test_frames}프레임 한도 도달.")
                running = False

    except KeyboardInterrupt:
        pass
    finally:
        display.quit()
        for vs in sources.values():
            vs.release()
        print("[main] 종료.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default=DEFAULT_CONFIG)
    parser.add_argument("--source-0",  dest="source_0", default=None)
    parser.add_argument("--source-1",  dest="source_1", default=None)
    parser.add_argument("--test",      type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    config["_path"] = args.config

    overrides = {}
    if args.source_0: overrides[0] = args.source_0
    if args.source_1: overrides[1] = args.source_1
    if overrides:
        apply_source_overrides(config, overrides)

    run(config, test_frames=args.test)


if __name__ == "__main__":
    main()
