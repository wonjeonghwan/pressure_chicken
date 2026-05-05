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

# 세그멘테이션 마스크 오버레이 표시 여부 (M 키로 토글)
_show_mask = True

_CV2_KEY_CAM = ord('c')  # 카메라 전환 키 (cv2 창 포커스 시)


def draw_preview(
    frames: dict,
    burners_cfg: list[dict],
    registry: BurnerRegistry,
    processor: FrameProcessor | None = None,
    config: dict | None = None,
) -> int:
    """소스별 영상에 ROI + 감지 결과 + optical flow 오버레이를 cv2 창에 표시."""
    rms_thr = (config or {}).get("optical_flow", {}).get("rms_threshold", 0.5)

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
            bid = b["id"]

            # 1. 캘리브레이션 ROI 경계 (옅은 회색)
            x, y, w, h = roi
            cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

            if not processor:
                continue

            bsm = registry.get(bid)
            if not bsm:
                continue
            r, g, b_val = bsm.color
            color_bgr = (b_val, g, r)

            # 2. 밥솥 bbox (상태 색상)
            if bid in processor.last_matched_boxes:
                bx1, by1, bx2, by2 = processor.last_matched_boxes[bid]
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), color_bgr, 2)
                cv2.putText(vis, f"Pot {bid}", (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

            # 3. 딸랑이 — seg 마스크 오버레이 + bbox + IoU + 점수 바
            if bid in processor.last_weight_boxes:
                wx1, wy1, wx2, wy2 = processor.last_weight_boxes[bid]

                # 3-a. 세그멘테이션 마스크 반투명 오버레이 (M 키로 토글)
                if _show_mask and bid in processor.last_mask_xys:
                    pts = processor.last_mask_xys[bid].astype(np.int32)
                    overlay = vis.copy()
                    cv2.fillPoly(overlay, [pts], (0, 230, 120))   # 초록-민트
                    cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
                    cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 150), thickness=2)

                if _show_mask:
                    # 3-b. 딸랑이 bbox (민트색)
                    cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (0, 255, 200), 2)

                    # 3-c. optical flow RMS 수치 (bbox 위쪽)
                    smoothed_rms = bsm.current_angle   # EMA 후
                    raw_rms      = bsm.angle_deviation  # EMA 전 raw
                    if smoothed_rms is not None:
                        rms_color = (0, 80, 255) if smoothed_rms >= rms_thr else (0, 220, 0)
                        # mask 사용 여부 표시 (M=mask, B=bbox fallback)
                        mask_flag = "M" if bid in processor.last_mask_xys else "B"
                        raw_str = f"({raw_rms:.2f})" if raw_rms is not None else ""
                        label = f"RMS:{smoothed_rms:.2f}{raw_str} [{mask_flag}]"
                        cv2.putText(vis, label, (wx1, wy1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, rms_color, 1)

                    # 3-d. 점수 바 (bbox 아래쪽)
                    score = bsm.vibration_score
                    bar_x, bar_y = wx1, wy2 + 4
                    bar_w = wx2 - wx1
                    bar_h = 6
                    cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                                  (60, 60, 60), -1)
                    filled = int(bar_w * min(score, 1.0))
                    bar_color = (0, 200, 0) if score < 1.0 else (0, 60, 255)
                    if filled > 0:
                        cv2.rectangle(vis, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                                      bar_color, -1)

            # 4. 무게중심 점 (딸랑이 중심) — 마스크 ON일 때만 표시
            if _show_mask and bid in processor.last_centroids:
                cx, cy = processor.last_centroids[bid]
                cv2.circle(vis, (cx, cy), 5, (0, 255, 150), -1)
                cv2.circle(vis, (cx, cy), 6, (0, 0, 0), 1)

        # 마스크 토글 상태 표시 (우상단)
        mask_label = "Mask: ON  [M]" if _show_mask else "Mask: OFF [M]"
        mask_color = (0, 220, 80) if _show_mask else (80, 80, 80)
        cv2.putText(vis, mask_label, (vis.shape[1] - 160, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, mask_color, 1, cv2.LINE_AA)

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

    _FRAME_INTERVAL = 1 / 15   # 영상 디코딩/표시 최대 15fps (4K 부하 절감)

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
                
        # 파일 소스: 실시간 카메라와 동일한 시간축이 되도록 프레임 스킵 계산
        # skip = round(video_fps / target_fps) → read() 1회당 video에서 skip 프레임 소비
        if sc.get("type") == "file":
            target_fps = 1 / _FRAME_INTERVAL  # main 루프 기준 처리 fps (15)
            skip = max(1, round(vs.fps / target_fps))
            sc["_skip_frames"] = skip
            print(f"[main] 파일 소스 {sc['id']}: video={vs.fps:.1f}fps, target={target_fps:.0f}fps → skip={skip}")

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

    # 4) FrameProcessor (Phase 1+2 통합)
    processor = FrameProcessor(sources, burners_cfg, registry, detector, config)

    # 5) UI
    display = UIDisplay(ui_cfg, registry, burner_meta, model_missing=detector.model_missing)
    display.init()

    print("[main] 시작. Q 로 종료.")
    if detector.model_missing:
        print("[main] 테스트 모드: 자동 감지 비활성화. ▶ 버튼으로 수동 시작 가능.")

    import time as _time
    _last_frame      = 0.0
    _last_detect     = 0.0
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
                cv2_key = draw_preview(current_frames, burners_cfg, registry, processor, config)
                
                # OpenCV 키 입력 (대소문자 처리)
                if cv2_key != -1:
                    key_code = cv2_key & 0xFF
                    if key_code == ord('c') or key_code == ord('C'):
                        trigger_camera_switch()
                    elif key_code == ord('m') or key_code == ord('M'):
                        global _show_mask
                        _show_mask = not _show_mask
                        print(f"[main] 마스크 오버레이: {'ON' if _show_mask else 'OFF'}")

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
