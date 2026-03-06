"""
ROI 캘리브레이션

각 화구의 ROI(감지 영역)를 마우스 드래그로 지정하고
config/store_config.json 에 저장한다.

실행:
    uv run python main.py --calibrate
    uv run python main.py --calibrate --config config/store_001.json
"""

import json
import os
import sys

import cv2
import numpy as np

from sources.video_source import VideoSource


# ── 마우스 콜백 상태 ────────────────────────────────────────────────────────
_drag_start: tuple[int, int] = (0, 0)
_drag_end:   tuple[int, int] = (0, 0)
_dragging = False
_drag_done = False


def _mouse_cb(event, x, y, flags, param):
    global _drag_start, _drag_end, _dragging, _drag_done
    if event == cv2.EVENT_LBUTTONDOWN:
        _drag_start = (x, y)
        _drag_end   = (x, y)
        _dragging   = True
        _drag_done  = False
    elif event == cv2.EVENT_MOUSEMOVE and _dragging:
        _drag_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drag_end  = (x, y)
        _dragging  = False
        _drag_done = True


def _to_roi(p1: tuple, p2: tuple) -> list[int]:
    """두 점 → [x, y, w, h] (항상 양수 크기)"""
    x = min(p1[0], p2[0])
    y = min(p1[1], p2[1])
    w = abs(p2[0] - p1[0])
    h = abs(p2[1] - p1[1])
    return [x, y, w, h]


def run_calibration(config_path: str) -> None:
    global _drag_done, _drag_start, _drag_end

    if not os.path.exists(config_path):
        print(f"[캘리브레이션] config 파일 없음: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # 소스 열기
    sources: dict[int, VideoSource] = {}
    for sc in config["sources"]:
        vs = VideoSource(sc)
        vs.open()
        sources[sc["id"]] = vs

    burners: list[dict] = list(config["burners"])

    print("=" * 56)
    print("ROI 캘리브레이션")
    print("  드래그  : 화구 영역 선택")
    print("  Space/Enter : 확정 후 다음 화구")
    print("  r       : 현재 화구 다시 그리기")
    print("  s       : 현재 화구 건너뛰기 (기존값 유지)")
    print("  ESC     : 중단 (저장 안 함)")
    print("=" * 56)

    win = "ROI Calibration  |  drag=select  Space=confirm  r=redo  s=skip  ESC=abort"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    for i, burner in enumerate(burners):
        bid    = burner["id"]
        src_id = burner["source_id"]
        vs     = sources.get(src_id)

        _drag_done  = False
        _drag_start = (0, 0)
        _drag_end   = (0, 0)

        print(f"\n[{i+1}/{len(burners)}] 화구 {bid}번  (현재 roi: {burner.get('roi', '없음')})")

        while True:
            if vs is not None:
                ret, frame = vs.read()
                display = frame.copy() if (ret and frame is not None) else np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                display = np.zeros((480, 640, 3), dtype=np.uint8)

            # 기존 ROI 초록 점선
            if burner.get("roi"):
                rx, ry, rw, rh = burner["roi"]
                cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0, 200, 0), 1)

            # 드래그 중인 ROI 주황색 실선
            if _dragging or _drag_done:
                cv2.rectangle(display, _drag_start, _drag_end, (0, 140, 255), 2)

            # 안내 오버레이
            h_img = display.shape[0]
            label = f"Burner #{bid}  ({i+1}/{len(burners)})"
            cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(display, "Space:OK  r:redo  s:skip  ESC:abort",
                        (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.imshow(win, display)
            key = cv2.waitKey(30) & 0xFF

            if key == 27:  # ESC
                print("[캘리브레이션] 중단. 저장하지 않습니다.")
                cv2.destroyAllWindows()
                for s in sources.values():
                    s.release()
                return

            if key in (13, 32):  # Enter / Space
                if _drag_done:
                    roi = _to_roi(_drag_start, _drag_end)
                    if roi[2] > 5 and roi[3] > 5:
                        burners[i] = {**burner, "roi": roi}
                        print(f"  ROI 확정: {roi}")
                        break
                    else:
                        print("  너무 작습니다. 다시 드래그하세요.")
                else:
                    print("  먼저 드래그해서 영역을 선택하세요.")

            if key == ord("r"):
                _drag_done = False
                print("  다시 그리기.")

            if key == ord("s"):
                print(f"  건너뜀 (roi 유지: {burner.get('roi', '없음')})")
                break

    cv2.destroyAllWindows()
    for s in sources.values():
        s.release()

    # 저장
    config["burners"] = burners
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n[캘리브레이션 완료] {config_path} 저장됨.")
