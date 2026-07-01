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
from sources.camera_utils import save_config, switch_camera, find_unused_index
from sources.mf_enum import list_cameras, find_unclaimed
from core.state_machine import BurnerRegistry
from core.detector import BurnerDetector
from core.frame_processor import FrameProcessor
from core.data_logger import DataLogger
from core.frame_saver import FrameSaver
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

    # 시작 시 PnP 카메라 목록 — 연결 확인 및 경고용 (열기는 하지 않음)
    _pnp_locations = {c.location for c in list_cameras()} if sys.platform == "win32" else set()

    for sc in sources_cfg:
        vs = VideoSource(sc)
        vs.open()

        if vs.failed and sc.get("type", "camera") == "camera":
            stored_loc = sc.get("device_location", "")
            if stored_loc and _pnp_locations and stored_loc not in _pnp_locations:
                print(f"[main] 소스 {sc['id']}: 저장된 카메라(포트 {stored_loc})가 현재 연결되어 있지 않습니다.")
            else:
                print(f"[main] 소스 {sc['id']} 열기 실패 (index={sc.get('index')}). 빈 소스로 동작합니다.")

        if sc.get("type") == "file":
            skip = max(1, round(vs.fps / target_fps))
            sc["_skip_frames"] = skip
            print(f"[main] 파일 소스 {sc['id']}: skip={skip}")
        elif not vs.failed:
            vs.start_capture(fps=target_fps)

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
            b.get("countdown_second", 270),
            b.get("done_first_timeout", 120),
            b.get("pot_absent_threshold", 30),
        )
        burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}

    # 3) Detector
    detector = BurnerDetector(weights, confidence)

    # 4) Processor
    processor = FrameProcessor(sources, burners_cfg, registry, detector, config)

    # 5) Logger
    logger = DataLogger(log_dir=config.get("log_dir", "logs"))

    # 6-a) FrameSaver — 학습 데이터용 주기 스냅샷 (메인 루프 영향 없음)
    saver_cfg = config.get("frame_saver", {})
    frame_saver: FrameSaver | None = None
    if saver_cfg.get("enabled", True):
        frame_saver = FrameSaver(
            sources=sources,
            log_dir=config.get("log_dir", "logs"),
            interval_sec=int(saver_cfg.get("interval_sec", 7200)),
            jpeg_quality=int(saver_cfg.get("jpeg_quality", 92)),
        )
        frame_saver.start()

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
    logger.log_camera_info(sources)

    # 6) UI
    display = UIDisplay(
        ui_cfg=ui_cfg,
        registry=registry,
        burner_meta=burner_meta,
        config_data=config,
        config_path=config.get("_path"),
        model_missing=detector.model_missing,
        logger=logger,
    )
    display.init()

    print("[main] 대시보드 시작. Q 로 종료.")

    import time as _time
    _last_detect     = 0.0
    _DETECT_INTERVAL = _FRAME_INTERVAL

    current_frames = {}
    frame_count = 0
    running = True
    
    _last_cam_switch = 0.0
    _CAM_SWITCH_COOLDOWN = 0.5

    def trigger_camera_switch(selected_src_id=None):
        nonlocal _last_cam_switch
        now = _time.monotonic()
        if now - _last_cam_switch < _CAM_SWITCH_COOLDOWN:
            return
        _last_cam_switch = now
        targets = [selected_src_id] if selected_src_id in cam_indices else list(cam_indices)
        for src_id in targets:
            cam_indices[src_id] = switch_camera(
                sources, src_id, cam_indices[src_id],
                config=config, config_path=config.get("_path")
            )
            # switch_camera() 가 새 VideoSource를 sources[src_id]에 넣음 → 스레드 시작
            new_vs = sources.get(src_id)
            if new_vs and not new_vs.failed:
                new_vs.start_capture(fps=target_fps)
            
    def handle_config_reloaded(new_cfg: dict):
        nonlocal burners_cfg, registry, processor
        print("[main] 핫 리로드 실행: 화구 설정이 메모리에 즉각 반영됩니다.")
        burners_cfg = new_cfg.get("burners", [])
        
        new_registry = BurnerRegistry()
        new_meta = {}
        for b in burners_cfg:
            new_registry.add(b["id"], b.get("countdown_first", 720), b.get("countdown_second", 270), b.get("done_first_timeout", 120), b.get("pot_absent_threshold", 30))
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
        """
        카메라 추가 — PnP LocationInfo 기반 안정 ID 매핑.

        1. PnP로 미등록 물리 카메라(location)를 파악
        2. 미사용 MSMF index에서 실제 프레임이 나오는 것을 탐색
        3. (미등록 카메라 1대 ↔ 미사용 index 1개) → 동일 물리 장치로 간주하고 매핑 저장
        """
        used_indices  = {sc.get("index") for sc in config.get("sources", [])
                         if sc.get("type", "camera") == "camera"}
        used_locations = {sc.get("device_location", "") for sc in config.get("sources", [])
                          if sc.get("type", "camera") == "camera" and sc.get("device_location")}
        used_ids = {sc["id"] for sc in config.get("sources", [])}
        next_src_id = max(used_ids) + 1 if used_ids else 0

        # ── PnP로 미등록 물리 카메라 확인 ─────────────────────────────────────
        unclaimed = find_unclaimed(used_locations)
        if not unclaimed:
            # PnP 열거 실패(비-Windows 등)이면 연결 카메라 수로 판단
            all_cams = list_cameras()
            if not all_cams:
                # 비-Windows: PnP 없이 index 탐색만
                pass
            else:
                print(f"[main] 추가할 카메라 없음 — 미등록 장치 없음 (연결 {len(all_cams)}대, 등록 {len(used_locations)}대)")
                return False

        # ── 미사용 MSMF index 탐색 (기존 핸들 건드리지 않음) ─────────────────
        result = find_unused_index(used_indices)
        if result is None:
            print(f"[main] 추가할 카메라 없음 — 미사용 index 없음 (사용중: {sorted(used_indices)})")
            return False

        new_index, vs = result

        # ── 물리 카메라 ↔ index 매핑 저장 ────────────────────────────────────
        device_location = unclaimed[0].location if unclaimed else ""
        device_name     = unclaimed[0].name     if unclaimed else "Camera"
        if len(unclaimed) > 1:
            print(f"[main] 경고: 미등록 카메라 {len(unclaimed)}대 — 첫 번째를 index {new_index}에 매핑")

        new_sc = {
            "id":              next_src_id,
            "type":            "camera",
            "index":           new_index,
            "label":           f"Cam-{next_src_id}",
            "device_location": device_location,
        }
        vs.start_capture(fps=target_fps)
        sources[next_src_id]    = vs
        cam_indices[next_src_id] = new_index
        config.setdefault("sources", []).append(new_sc)
        save_config(config.get("_path"), config)
        handle_config_reloaded(config)
        print(f"[main] 카메라 추가: src#{next_src_id} = index {new_index}  location={device_location!r}")
        return True

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
                if not display.calibration_mode:
                    if now - _last_detect >= _DETECT_INTERVAL:
                        # 카메라 소스: get_latest() 비블로킹 — 캡처 스레드 버퍼에서 즉시 반환
                        # 파일 소스: read() 동기 — 기존 방식 유지
                        current_frames = processor.read_frames()
                        processor.detect_and_update()
                        logger.update(registry, sources)
                        frame_count += 1
                        _last_detect = now
                else:
                    # 캘리브레이션 모드: 감지 없이 화면만 갱신
                    current_frames = processor.read_frames()

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
        if frame_saver:
            frame_saver.stop()
        logger.close()
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
