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

import cv2
import numpy as np
import pygame

from sources.video_source import VideoSource
from sources.camera_utils import save_config, switch_camera
from core.state_machine import BurnerRegistry
from core.detector import BurnerDetector
from core.frame_processor import FrameProcessor
from ui.ui_display import UIDisplay

DEFAULT_CONFIG = "config/store_config.json"

# 미리보기 창 축소 비율 (1.0 = 원본, 0.4 = 40%)
_PREVIEW_SCALE = 1.0


_CV2_KEY_CAM = ord('c')  # 카메라 전환 키 (cv2 창 포커스 시)


def draw_preview(
    frames: dict,
    burners_cfg: list[dict],
    registry: BurnerRegistry,
    processor: FrameProcessor | None = None,
    motion_cfg: dict | None = None,
) -> int:
    """소스별 영상에 ROI 박스 + 화구번호 + 상태 오버레이 후 cv2 창에 표시."""
    src_burners: dict[int, list[dict]] = {}
    for b in burners_cfg:
        src_burners.setdefault(b["source_id"], []).append(b)

    for src_id, frame in frames.items():
        if frame is None:
            continue
        vis = frame.copy()
        for b in src_burners.get(src_id, []):
            roi = b.get("roi")
            if not roi:
                continue

            # 1. 고정 ROI (캘리브레이션 영역 - 옅은 회색)
            x, y, w, h = roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

            # 2. 실시간 YOLO 매칭 결과 (바디 & 딸랑이)
            if processor:
                # 바디 박스 (상태별 색상)
                bsm = registry.get(b["id"])
                if bsm:
                    r, g, b_val = bsm.color
                    color_bgr = (b_val, g, r) # RGB to BGR
                    
                    if b["id"] in processor.last_matched_boxes:
                        bx1, by1, bx2, by2 = processor.last_matched_boxes[b["id"]]
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color_bgr, 2)
                        cv2.putText(vis, f"Pot {b['id']}", (bx1, by1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

                # 딸랑이 박스 (노란색) + dark_threshold 마스크 인셋
                if b["id"] in processor.last_weight_boxes:
                    wx1, wy1, wx2, wy2 = processor.last_weight_boxes[b["id"]]
                    cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2)
                    cv2.putText(vis, f"Whistle {b['id']}", (wx1, wy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    # dark_threshold 마스크 인셋 (딸랑이 박스 오른쪽 옆에 표시)
                    roi = vis[wy1:wy2, wx1:wx2]
                    if roi.size > 0:
                        dark_thr = (motion_cfg or {}).get("dark_threshold", 50)
                        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                        mask_bin = (gray_roi < dark_thr).astype("uint8") * 255
                        mask_bgr = cv2.cvtColor(mask_bin, cv2.COLOR_GRAY2BGR)
                        # 인셋 크기: 딸랑이 박스와 동일, 2배 확대
                        scale   = max(1, 64 // max(1, wy2 - wy1))
                        iw = (wx2 - wx1) * scale
                        ih = (wy2 - wy1) * scale
                        inset = cv2.resize(mask_bgr, (iw, ih), interpolation=cv2.INTER_NEAREST)
                        # 인셋 위치: 딸랑이 박스 오른쪽
                        ix1 = min(wx2 + 4, vis.shape[1] - iw)
                        iy1 = max(0, wy1)
                        iy2 = iy1 + ih
                        ix2 = ix1 + iw
                        if ix2 <= vis.shape[1] and iy2 <= vis.shape[0]:
                            vis[iy1:iy2, ix1:ix2] = inset
                            cv2.rectangle(vis, (ix1, iy1), (ix2, iy2), (0, 255, 255), 1)

                # 무게중심 시각화 (초록 원)
                if b["id"] in processor.last_centroids:
                    cx, cy = processor.last_centroids[b["id"]]
                    cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
                    cv2.circle(vis, (cx, cy), 7, (0, 0, 0), 1)  # 외곽선

                # 키포인트 시각화 (빨간색 원 + 연결선)
                if b["id"] in processor.last_keypoints:
                    kp_t, kp_b = processor.last_keypoints[b["id"]]
                    pt_top = (int(kp_t[0]), int(kp_t[1]))
                    pt_bot = (int(kp_b[0]), int(kp_b[1]))
                    cv2.circle(vis, pt_top, 5, (0, 0, 255), -1)   # 빨간 원 (top)
                    cv2.circle(vis, pt_bot, 5, (0, 80, 255), -1)  # 주황 원 (bot)
                    cv2.line(vis, pt_top, pt_bot, (0, 0, 255), 2)  # 연결선

        # _PREVIEW_SCALE 적용
        h_img, w_img = vis.shape[:2]
        if _PREVIEW_SCALE != 1.0:
            vis = cv2.resize(vis, (int(w_img * _PREVIEW_SCALE), int(h_img * _PREVIEW_SCALE)))
        
        window_name = f"Camera {src_id}"
        
        # 1. 창이 없으면 먼저 만듦 (macOS Cocoa는 미존재 창에 getWindowProperty 호출 시 예외 발생)
        try:
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = -1
        if visible < 1:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
        cv2.imshow(window_name, vis)
        
        # 2. 강제 윈도우 스냅 크기 조절 (운영체제 창 테두리)
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and rect[2] > 0 and rect[3] > 0:
            win_w, win_h = rect[2], rect[3]
            src_h, src_w = vis.shape[:2]
            
            current_ratio = win_w / win_h
            target_ratio = src_w / src_h
            
            # 사용자가 창 비율을 왜곡했을 경우 (5% 오차 허용) 강제로 세로축을 가로축에 맞춰 교정
            if abs(current_ratio - target_ratio) > 0.05:
                target_h = int(win_w / target_ratio)
                cv2.resizeWindow(window_name, win_w, target_h)

    return cv2.waitKeyEx(1)


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
    """Override source config with CLI arguments (e.g., video files)"""
    sources = config.get("sources", [])
    for sc in sources:
        sid = sc.get("id")
        if sid in overrides:
            sc["type"] = "file"
            sc["path"] = overrides[sid]


# Removed local save_config and _switch_camera as they are now in camera_utils.py


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
    cam_indices: dict[int, int] = {}   # source_id → 현재 카메라 인덱스
    
    config_path = config.get("_path") # pass _path if needed or use global

    for sc in sources_cfg:
        vs = VideoSource(sc)
        vs.open()
        
        # 카메라 오프닝 실패 시 사용 가능한 카메라 자동 탐색 (Fallback)
        if vs.failed and sc.get("type", "camera") == "camera":
            print(f"[main] 소스 {sc['id']} 로드 실패. 사용 가능한 다른 카메라를 찾습니다...")
            available = VideoSource.find_available_cameras()
            if available:
                new_idx = available[0]
                print(f"[main] -> 대체 카메라 발견 (index: {new_idx}). 연결을 시도합니다.")
                sc["index"] = new_idx
                vs = VideoSource(sc)
                vs.open()
                
        sources[sc["id"]] = vs
        if sc.get("type", "camera") == "camera":
            cam_indices[sc["id"]] = sc.get("index", 0)

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

    import time as _time
    _last_frame      = 0.0
    _last_detect     = 0.0
    _FRAME_INTERVAL  = 1 / 15   # 영상 디코딩/표시 최대 15fps (4K 부하 절감)
    _DETECT_INTERVAL = 1 / 15   # YOLO 감지 초당 15회 (영상읽기와 동일)

    # FPS 측정
    _fps_video_count  = 0
    _fps_detect_count = 0
    _fps_report_at    = _time.monotonic() + 5.0   # 5초마다 출력

    current_frames: dict = {}
    frame_count = 0
    running     = True
    
    # 카메라 전환 쿨다운 (중복 입력 방지)
    _last_cam_switch = 0.0
    _CAM_SWITCH_COOLDOWN = 0.5

    try:
        while running:
            now = _time.monotonic()
            
            # 1) 공통 카메라 전환 함수
            def trigger_camera_switch():
                nonlocal _last_cam_switch
                if now - _last_cam_switch < _CAM_SWITCH_COOLDOWN:
                    return
                _last_cam_switch = now
                for src_id in list(cam_indices):
                    cam_indices[src_id] = switch_camera(
                        sources, src_id, cam_indices[src_id],
                        config=config, config_path=config.get("_path")
                    )

            # 2) Pygame 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_c:
                    trigger_camera_switch()
                elif display.handle_event(event):
                    running = False

            # 3) 영상 읽기 + 미리보기: 15fps
            if now - _last_frame >= _FRAME_INTERVAL:
                current_frames = processor.read_frames()
                cv2_key = draw_preview(current_frames, burners_cfg, registry, processor, motion_cfg)
                
                # OpenCV 키 입력 (대소문자 'c'/'C' 처리)
                if cv2_key != -1:
                    key_code = cv2_key & 0xFF
                    if key_code == ord('c') or key_code == ord('C'):
                        trigger_camera_switch()

                _last_frame = now
                _fps_video_count += 1

            # YOLO 감지: 초당 3회 (배치 추론으로 화구 수 무관 1회 호출)
            if now - _last_detect >= _DETECT_INTERVAL:
                processor.detect_and_update()
                _last_detect = now
                _fps_detect_count += 1

            display.render()

            # FPS 출력 (5초마다)
            if now >= _fps_report_at:
                print(f"[FPS] 영상읽기 {_fps_video_count/5:.1f}/s  |  YOLO감지 {_fps_detect_count/5:.1f}/s")
                _fps_video_count  = 0
                _fps_detect_count = 0
                _fps_report_at    = now + 5.0

            frame_count += 1
            if test_frames > 0 and frame_count >= test_frames:
                print(f"[main] 테스트 {test_frames}프레임 완료. 정상 종료.")
                running = False

    except KeyboardInterrupt:
        print("[main] Ctrl+C 감지. 종료 중...")
    finally:
        # 창 종료 / 예외 / Ctrl+C 모든 경우에 반드시 정리
        display.quit()
        cv2.destroyAllWindows()
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

    config = load_config(args.config)
    config["_path"] = args.config  # 저장을 위한 경로 기록

    overrides: dict[int, str] = {}
    if args.source_0:
        overrides[0] = args.source_0
    if args.source_1:
        overrides[1] = args.source_1
    if overrides:
        apply_source_overrides(config, overrides)

    if args.calibrate:
        from calibration import run_calibration
        run_calibration(args.config, config)
        return

    run(config, test_frames=args.test)


if __name__ == "__main__":
    main()
