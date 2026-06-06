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


def _resolve_target_fps(ui_cfg: dict, n_cams: int) -> float:
    """target_fps: ui.target_fps 명시 > 카메라 수 기반 기본값 (1~2:15, 3~6:10, 7+:8)."""
    explicit = ui_cfg.get("target_fps")
    if explicit:
        return float(explicit)
    if n_cams <= 2:
        return 15.0
    if n_cams <= 6:
        return 10.0
    return 8.0


def run(config: dict, test_frames: int = 0, screenshot_path: str | None = None) -> None:
    sources_cfg  = config["sources"]
    burners_cfg  = config.get("burners", [])
    ui_cfg       = config.get("ui", {})
    model_cfg    = config.get("model", {})

    weights    = model_cfg.get("weights", "models/pot_detector.pt")
    confidence = model_cfg.get("confidence", 0.5)

    target_fps = _resolve_target_fps(ui_cfg, len(sources_cfg))
    _FRAME_INTERVAL = 1.0 / target_fps
    print(f"[main] target_fps={target_fps} (cams={len(sources_cfg)})")

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
            b.get("done_first_timeout", 600),
            b.get("pot_absent_threshold", 60),
        )
        burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}

    # 3) Detector
    detector = BurnerDetector(weights, confidence)

    # 4) Processor
    processor = FrameProcessor(sources, burners_cfg, registry, detector, config)

    def print_load_diagnostics(proc: FrameProcessor) -> None:
        """카메라별 ROI 합집합 면적 + 화구 수를 콘솔에 출력 — 매장 셋업 진단용."""
        # 각 source의 resize 또는 기본 해상도 추정
        res_by_src: dict[int, tuple[int, int]] = {}
        for sc in config.get("sources", []):
            r = sc.get("resize")
            if r and len(r) == 2:
                res_by_src[sc["id"]] = (int(r[0]), int(r[1]))
            else:
                vs = sources.get(sc["id"])
                if vs:
                    w, h = vs.frame_size
                    if w > 0 and h > 0:
                        res_by_src[sc["id"]] = (w, h)

        # 카메라별 coverage 계산
        for sc in config.get("sources", []):
            sid = sc["id"]
            cov = proc.estimate_roi_coverage(res_by_src.get(sid))
            info = cov.get(sid, {"n_burners": 0, "coverage": None})
            n_b = info["n_burners"]
            c = info["coverage"]
            if n_b == 0:
                print(f"[diag] Cam-{sid}: 화구 미할당")
                continue
            if c is None:
                print(f"[diag] Cam-{sid}: burners={n_b}, 해상도 미상 — 캘리브 후 확인")
                continue
            pct = c * 100
            tag = "✓ 가벼움" if pct < 40 else ("⚠ 적정" if pct < 65 else "⚠⚠ 카메라 추가 검토")
            print(f"[diag] Cam-{sid}: burners={n_b}, ROI 합집합={pct:.1f}%  → {tag}")

    print_load_diagnostics(processor)

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
    _DETECT_INTERVAL = _FRAME_INTERVAL

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
            
    def handle_config_reloaded(new_cfg: dict):
        nonlocal burners_cfg, registry, processor
        print("[main] 핫 리로드 실행: 화구 설정이 메모리에 즉각 반영됩니다.")
        burners_cfg = new_cfg.get("burners", [])
        
        new_registry = BurnerRegistry()
        new_meta = {}
        for b in burners_cfg:
            new_registry.add(b["id"], b.get("countdown_first", 720), b.get("countdown_second", 300), b.get("done_first_timeout", 600), b.get("pot_absent_threshold", 60))
            new_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}
            
        display._registry = new_registry
        display._meta = new_meta
        display._card_rects.clear()
        
        new_processor = FrameProcessor(sources, burners_cfg, new_registry, detector, config)
        registry = new_registry
        processor = new_processor
        print_load_diagnostics(processor)

    display.on_camera_switch = trigger_camera_switch
    display.on_config_reloaded = handle_config_reloaded

    def add_camera() -> bool:
        """안전한 카메라 추가 — 이미 열린 카메라 핸들을 건드리지 않음.

        기존 구현은 `VideoSource.find_available_cameras`를 호출해 0~9를 전부 열고/닫았는데,
        이 과정에서 이미 메인 프로그램이 잡고 있던 카메라 핸들이 DSHOW 백엔드에 의해
        강제 회수되며 'frame 없음' 상태로 빠지는 버그가 있었음.

        개선: 사용 중이 아닌 인덱스만 한 번씩 직접 시도. 이미 열린 카메라는 건드리지 않음.
        """
        used_indices = {sc.get("index") for sc in config.get("sources", []) if sc.get("type", "camera") == "camera"}
        used_ids = {sc["id"] for sc in config.get("sources", [])}
        next_src_id = max(used_ids) + 1 if used_ids else 0

        for candidate in range(10):
            if candidate in used_indices:
                continue
            new_sc = {"id": next_src_id, "type": "camera", "index": candidate, "label": f"Cam-{next_src_id}"}
            vs = VideoSource(new_sc)
            vs.open()
            if vs.failed:
                vs.release()
                continue
            # 첫 frame까지 받아봐야 실제 사용 가능한 인덱스. 일부 가상 카메라는 open만 성공
            ok, _frame = vs.read()
            if not ok:
                vs.release()
                continue
            sources[next_src_id] = vs
            cam_indices[next_src_id] = candidate
            config.setdefault("sources", []).append(new_sc)
            save_config(config.get("_path"), config)
            handle_config_reloaded(config)
            print(f"[main] 카메라 추가됨: src#{next_src_id} = index {candidate}")
            return True

        print(f"[main] 추가 가능한 카메라 인덱스 없음 (사용중: {sorted(used_indices)})")
        return False

    def remove_camera(source_id: int) -> bool:
        """지정된 source_id의 카메라 제거 + 묶인 화구 함께 제거 + 핫리로드."""
        srcs = config.get("sources", [])
        idx = next((i for i, sc in enumerate(srcs) if sc["id"] == source_id), None)
        if idx is None:
            print(f"[main] Cam-{source_id} 없음 — 제거 스킵")
            return False
        # 그 카메라에 묶인 화구도 제거
        n_removed_burners = sum(1 for b in config.get("burners", []) if b.get("source_id") == source_id)
        config["burners"] = [b for b in config.get("burners", []) if b.get("source_id") != source_id]
        vs = sources.pop(source_id, None)
        if vs:
            vs.release()
        cam_indices.pop(source_id, None)
        srcs.pop(idx)
        save_config(config.get("_path"), config)
        handle_config_reloaded(config)
        print(f"[main] 카메라 제거됨: src#{source_id} (화구 {n_removed_burners}개 함께 제거)")
        return True

    display.on_camera_add = add_camera
    display.on_camera_remove = remove_camera



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
                if screenshot_path:
                    try:
                        pygame.image.save(display._screen, screenshot_path)
                        print(f"[main] 스크린샷 저장: {screenshot_path}")
                    except Exception as e:
                        print(f"[main] 스크린샷 실패: {e}")
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
    parser.add_argument("--screenshot", default=None, help="--test N과 함께 사용. N프레임 후 스크린샷 저장.")
    parser.add_argument("--headless",  action="store_true", help="SDL_VIDEODRIVER=dummy 강제 (GUI 없이 실행)")
    args = parser.parse_args()

    if args.headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"

    config = load_config(args.config)
    config["_path"] = args.config

    overrides = {}
    if args.source_0: overrides[0] = args.source_0
    if args.source_1: overrides[1] = args.source_1
    if overrides:
        apply_source_overrides(config, overrides)

    run(config, test_frames=args.test, screenshot_path=args.screenshot)


if __name__ == "__main__":
    main()
