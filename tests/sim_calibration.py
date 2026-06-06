"""
캘리브 자동 그룹화 시뮬레이션.

마우스 이벤트를 직접 주입해 4 카메라 환경에서 일부러 순서 뒤죽박죽으로
화구를 그린 뒤, 저장 시 자동 그룹화가 잘 되는지 검증.

실행: uv run python -X utf8 tests/sim_calibration.py
"""
import os
import sys
import json
import shutil

# 상위 경로 import 가능하도록
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# 헤드리스 강제
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame

from core.state_machine import BurnerRegistry
from ui.ui_display import UIDisplay


def main():
    # 1. config 로드 (원본 안 망가지게 임시 사본 사용)
    src = "config/examples/store_4cam_video.json"
    tmp = "config/_sim_calib_tmp.json"
    shutil.copy(src, tmp)
    with open(tmp, encoding="utf-8") as f:
        config = json.load(f)
    config["_path"] = tmp

    # 2. UIDisplay 셋업
    registry = BurnerRegistry()
    display = UIDisplay(
        ui_cfg=config.get("ui", {}),
        registry=registry,
        burner_meta={},
        config_data=config,
        config_path=tmp,
        model_missing=False,
    )
    display.init()

    # 3. 가짜 frames (각 카메라 1280×720 검정 BGR — 셀 좌표만 채우면 됨)
    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frames = {0: fake_frame.copy(), 1: fake_frame.copy(), 2: fake_frame.copy(), 3: fake_frame.copy()}

    # 4. render 한 번 → _cam_cells 채워짐 + 캘리브 모드 자동 진입 상태
    display.render(frames=frames, processor=None)
    print(f"\n[sim] 캘리브 모드: {display.calibration_mode}")
    print(f"[sim] _cam_cells 등록된 카메라: {sorted(display._cam_cells.keys())}")

    # 5. 각 카메라 셀의 중심점 계산 (드래그 시작점으로 사용)
    cell_centers = {}
    for sid, cell in display._cam_cells.items():
        r = cell["rect"]
        cell_centers[sid] = (r.x + r.w // 2, r.y + r.h // 2)
    print(f"[sim] 셀 중심: {cell_centers}\n")

    def drag(sid: int, dx: int = 0, dy: int = 0, w: int = 80, h: int = 80):
        """sid 카메라 안 (cx+dx, cy+dy) 위치에서 (w, h) 박스 드래그.

        화면 좌표 → 마우스 이벤트 → UIDisplay 핸들러 직접 호출.
        """
        cx, cy = cell_centers[sid]
        start = (cx + dx, cy + dy)
        end = (start[0] + w, start[1] + h)
        display._on_mouse_down(start)
        display._on_mouse_move(end)
        display._on_mouse_up(end)
        print(f"  Cam-{sid} 위에 드래그: {start} → {end}")

    # 6. 의도적으로 순서를 뒤죽박죽 — Cam-2부터 그리기
    print("[sim] 화구를 의도적으로 순서 섞어서 드래그:")
    drag(2, dx=-100, dy=-100)              # Cam-2 (그린 순서 1)
    drag(0, dx=-100, dy=-100)              # Cam-0 (그린 순서 2)
    drag(0, dx= 100, dy=-100)              # Cam-0 (그린 순서 3) — 같은 카메라 두 번째
    drag(3, dx=-100, dy=-100)              # Cam-3 (그린 순서 4)
    drag(1, dx=-100, dy=-100)              # Cam-1 (그린 순서 5)

    print(f"\n[sim] 저장 직전 임시 ID/순서:")
    for b in display._calib_burners:
        print(f"  ID={b['id']}  source_id={b['source_id']}  roi={b['roi']}")

    # 7. save_calibration() — 자동 그룹화 작동
    display.save_calibration()

    # 8. 저장 결과 확인
    with open(tmp, encoding="utf-8") as f:
        saved = json.load(f)
    print(f"\n[sim] 저장 후 (자동 그룹화):")
    for b in saved["burners"]:
        print(f"  ID={b['id']}  source_id={b['source_id']}  roi={b['roi']}")

    # 9. 검증 — ID가 source_id 오름차순 + 1부터 연속
    burners = saved["burners"]
    src_ids = [b["source_id"] for b in burners]
    ids = [b["id"] for b in burners]
    assert src_ids == sorted(src_ids), f"source_id 오름차순 실패: {src_ids}"
    assert ids == list(range(1, len(ids) + 1)), f"ID 연속 실패: {ids}"
    print("\n[sim] ✓ 자동 그룹화 검증 통과")
    print(f"        source_id 오름차순: {src_ids}")
    print(f"        ID 1부터 연속:      {ids}")

    # 10. 캘리브 후 상태 스크린샷
    display.start_calibration()  # 다시 캘리브 모드 진입해서 확정된 화구 표시
    display.render(frames=frames, processor=None)
    out = "docs/ux_screenshots/sim_calib_after_save.png"
    pygame.image.save(display._screen, out)
    print(f"\n[sim] 스크린샷 저장: {out}")

    # 11. 정리
    display.quit()
    if os.path.exists(tmp):
        os.remove(tmp)

    print("\n[sim] 완료\n")


if __name__ == "__main__":
    main()
