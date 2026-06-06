"""
카메라 선택 + 제거 시뮬레이션.

영상 영역 클릭 → 외곽 노란 스트로크 + 제거 버튼 라벨 변화 검증.
실행: uv run python -X utf8 tests/sim_camera_select.py
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame

from core.state_machine import BurnerRegistry
from ui.ui_display import UIDisplay


def main():
    src = "config/examples/store_4cam_video_with_burners.json"
    tmp = "config/_sim_select_tmp.json"
    shutil.copy(src, tmp)
    with open(tmp, encoding="utf-8") as f:
        config = json.load(f)
    config["_path"] = tmp

    # registry 채우기 — 8 burners
    registry = BurnerRegistry()
    burner_meta = {}
    for b in config["burners"]:
        registry.add(b["id"], b.get("countdown_first", 720), b.get("countdown_second", 300),
                     b.get("done_first_timeout", 600), b.get("pot_absent_threshold", 60))
        burner_meta[b["id"]] = {"grid_pos": b.get("grid_pos", [0, b["id"] - 1])}

    display = UIDisplay(
        ui_cfg=config.get("ui", {}),
        registry=registry,
        burner_meta=burner_meta,
        config_data=config,
        config_path=tmp,
        model_missing=False,
    )
    display.init()
    display.calibration_mode = False  # 일반 모드 강제 (burners 있으므로 자동 진입 X)

    fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frames = {sid: fake_frame.copy() for sid in (0, 1, 2, 3)}

    display.render(frames=frames, processor=None)
    print(f"\n[sim] 등록된 카메라 셀: {sorted(display._cam_cells.keys())}")

    # ─ 시나리오 1: Cam-2 셀 클릭 → 선택 ─
    cam2_rect = display._cam_cells[2]["rect"]
    center = (cam2_rect.x + cam2_rect.w // 2, cam2_rect.y + cam2_rect.h // 2)
    print(f"\n[sim] Cam-2 셀 중심 클릭: {center}")
    display._on_mouse_down(center)
    display._on_mouse_up(center)
    print(f"      → _selected_camera_id = {display._selected_camera_id}")
    assert display._selected_camera_id == 2

    display.render(frames=frames, processor=None)
    out1 = "docs/ux_screenshots/cam_select_cam2.png"
    pygame.image.save(display._screen, out1)
    print(f"      스크린샷: {out1}")

    # ─ 시나리오 2: 같은 셀 다시 클릭 → 선택 해제 ─
    print(f"\n[sim] Cam-2 다시 클릭 → 선택 해제")
    display._on_mouse_down(center)
    display._on_mouse_up(center)
    print(f"      → _selected_camera_id = {display._selected_camera_id}")
    assert display._selected_camera_id is None

    # ─ 시나리오 3: Cam-1 클릭 후 제거 버튼 클릭 ─
    cam1_rect = display._cam_cells[1]["rect"]
    cam1_center = (cam1_rect.x + cam1_rect.w // 2, cam1_rect.y + cam1_rect.h // 2)
    print(f"\n[sim] Cam-1 클릭 → 제거 버튼 클릭")
    display._on_mouse_down(cam1_center)
    display._on_mouse_up(cam1_center)
    assert display._selected_camera_id == 1

    # remove 콜백 가짜 연결
    removed_sid = []
    def fake_remove(sid):
        removed_sid.append(sid)
        return True
    display.on_camera_remove = fake_remove

    display.render(frames=frames, processor=None)  # _cam_remove_rect 채우기
    if display._cam_remove_rect:
        rc = display._cam_remove_rect.center
        display._on_mouse_down(rc)
        display._on_mouse_up(rc)
    print(f"      → 콜백 호출됨: source_id={removed_sid}")
    assert removed_sid == [1], f"제거 콜백 인자 오류: {removed_sid}"

    # ─ 시나리오 4: 선택 없이 제거 버튼 클릭 → 토스트 ─
    display._selected_camera_id = None
    removed_sid.clear()
    print(f"\n[sim] 선택 없이 제거 버튼 클릭")
    display.render(frames=frames, processor=None)
    rc = display._cam_remove_rect.center
    display._on_mouse_down(rc)
    display._on_mouse_up(rc)
    print(f"      → 콜백 호출 안 됨: {removed_sid == []}")
    print(f"      → 토스트 메시지: '{display._toast_msg}'")
    assert removed_sid == []
    assert display._toast_msg and "먼저 클릭" in display._toast_msg

    # 마지막 스크린샷 (Cam-3 선택 + 제거 버튼 라벨 확인)
    cam3_rect = display._cam_cells[3]["rect"]
    cam3_center = (cam3_rect.x + cam3_rect.w // 2, cam3_rect.y + cam3_rect.h // 2)
    display._on_mouse_down(cam3_center)
    display._on_mouse_up(cam3_center)
    display.render(frames=frames, processor=None)
    out2 = "docs/ux_screenshots/cam_select_cam3_with_remove_label.png"
    pygame.image.save(display._screen, out2)
    print(f"\n[sim] Cam-3 선택 스크린샷: {out2}")

    print("\n[sim] ✓ 모든 시나리오 통과")

    display.quit()
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == "__main__":
    main()
