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


def run_calibration(config_path: str, config: dict | None = None) -> None:
    """
    대화형 ROI 캘리브레이션.

    드래그로 화구 영역을 하나씩 지정하고 Space로 확정.
    ESC를 누르면 지정된 화구만 저장하고 종료.
    기존 store_config.json의 burners는 덮어씌워짐.

    조작:
      드래그        : 화구 영역 선택
      Space/Enter  : 현재 박스 화구로 확정 (다음 화구로)
      z            : 마지막 추가 화구 취소
      r            : 현재 드래그 다시 그리기
      ESC          : 저장 후 종료
    """
    global _drag_done, _drag_start, _drag_end

    if config is None:
        if not os.path.exists(config_path):
            print(f"[캘리브레이션] config 파일 없음: {config_path}")
            sys.exit(1)
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

    # 첫 번째 소스 열기
    sources_cfg = config.get("sources", [])
    if not sources_cfg:
        print("[캘리브레이션] sources 설정이 없습니다.")
        sys.exit(1)

    sources: dict[int, VideoSource] = {}
    for sc in sources_cfg:
        vs = VideoSource(sc)
        vs.open()
        sources[sc["id"]] = vs

    grid_cols = config.get("ui", {}).get("grid_cols", 6)

    burners: list[dict] = []   # 새로 정의할 화구 목록
    current_src = 0             # 현재 사용 중인 source_id

    print("=" * 60)
    print("ROI 캘리브레이션  (화구 위치를 직접 그려주세요)")
    print("  드래그         : 화구 영역 선택")
    print("  Space/Enter    : 확정 → 다음 화구")
    print("  z              : 마지막 화구 취소")
    print("  r              : 다시 그리기")
    print("  ESC            : 저장 후 종료")
    print("=" * 60)

    win = "ROI Calibration | drag=add  Space=confirm  z=undo  ESC=save&exit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    _drag_done  = False
    _drag_start = (0, 0)
    _drag_end   = (0, 0)

    vs = sources.get(current_src)

    while True:
        if vs is not None and not vs.failed:
            ret, frame = vs.read()
            base = frame.copy() if (ret and frame is not None) else np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            base = np.zeros((480, 640, 3), dtype=np.uint8)

        display = base.copy()

        # 확정된 화구 ROI (초록)
        for b in burners:
            rx, ry, rw, rh = b["roi"]
            cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 200, 0), 2)
            cv2.putText(display, str(b["id"]), (rx + 5, ry + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

        # 드래그 중 / 완료 (주황)
        if _dragging or _drag_done:
            cv2.rectangle(display, _drag_start, _drag_end, (0, 140, 255), 2)

        # 안내 텍스트
        next_id = len(burners) + 1
        h_img   = display.shape[0]
        cv2.putText(display, f"Burner #{next_id} | confirmed: {len(burners)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, "Space:confirm  z:undo  ESC:save&exit",
                    (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:  # ESC → 저장 후 종료
            break

        if key in (13, 32):  # Space / Enter → 확정
            if _drag_done:
                roi = _to_roi(_drag_start, _drag_end)
                if roi[2] > 5 and roi[3] > 5:
                    idx     = len(burners)
                    grid_pos = [idx // grid_cols, idx % grid_cols]
                    burners.append({
                        "id":              next_id,
                        "source_id":       current_src,
                        "countdown_first": 720,
                        "countdown_second": 300,
                        "grid_pos":        grid_pos,
                        "roi":             roi,
                    })
                    print(f"  화구 {next_id}번 확정: roi={roi}  grid={grid_pos}")
                    _drag_done  = False
                    _drag_start = (0, 0)
                    _drag_end   = (0, 0)
                else:
                    print("  너무 작습니다. 다시 드래그하세요.")
            else:
                print("  먼저 드래그해서 영역을 선택하세요.")

        elif key == ord("z"):  # undo
            if burners:
                removed = burners.pop()
                print(f"  화구 {removed['id']}번 취소.")
            _drag_done = False

        elif key == ord("r"):  # redo
            _drag_done = False
            print("  다시 그리기.")

    cv2.destroyAllWindows()
    for s in sources.values():
        s.release()

    if not burners:
        print("[캘리브레이션] 화구 미정의. 저장하지 않습니다.")
        return

    config["burners"] = burners
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n[캘리브레이션 완료] 화구 {len(burners)}개 저장 → {config_path}")
