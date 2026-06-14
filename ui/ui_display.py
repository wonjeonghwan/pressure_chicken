"""
pygame 타이머 UI — 통합 대시보드
브랜드 컬러: 길가옆에 누룽지 삼계탕 (Yellow / Black / White)
단일 윈도우 지원, In-App 캘리브레이션 지원.
"""

from __future__ import annotations

import time
import json
import os
import cv2
import numpy as np
import pygame

from core.state_machine import BurnerRegistry, BurnerState, _STEAMING

# ── 브랜드 및 테마 색상 ───────────────────────────────────────────
_C_BRAND       = (255, 192, 0)     # 길가옆에 메인 옐로우
_C_BRAND_HOVER = (255, 210, 50)
_C_BG          = (20,  20,  20)    # 전체 배경 (진한 흑색)
_C_PANEL       = (35,  35,  35)    # 패널 배경
_C_TEXT_LIGHT  = (240, 240, 240)   # 기본 텍스트 백색
_C_TEXT_DARK   = (20,  20,  20)    # 검정 텍스트 (옐로우 위)
_C_SELECTED    = (255, 192, 0)     # 선택 테두리
_C_CARD_BG     = (45,  45,  45)
_C_CARD_BORDER = (60,  60,  60)

_C_WARNING_BG  = (180, 40,  40)
_C_SUCCESS     = (40, 180, 60)
_C_CONNECTED   = (40, 200, 80)   # 연결 상태 LED — 초록
_C_NO_FRAME    = (220, 170, 30)  # 연결됐으나 프레임 없음 — 주황
_C_FAILED      = (200, 60, 60)   # 열기 실패 — 빨강

_PAD = 8
_RIGHT_PANEL_W = 400
_RESET_HOLD_S  = 1.0
_CELL_GAP      = 4               # 그리드 셀 간 여백

class UIDisplay:
    """단일 통합 Pygame 대시보드"""

    def __init__(
        self,
        ui_cfg: dict,
        registry: BurnerRegistry,
        burner_meta: dict,
        config_data: dict,
        config_path: str,
        model_missing: bool = True,
    ):
        self._cfg          = ui_cfg
        self._registry     = registry
        self._meta         = burner_meta
        self.config_data   = config_data
        self.config_path   = config_path
        self._model_missing = model_missing

        self._screen: pygame.Surface | None = None
        self._clock:  pygame.time.Clock | None = None
        self._fonts:  dict | None = None

        self._selected_id: int | None = None
        self._selected_camera_id: int | None = None  # 선택된 카메라 source_id
        self._reset_hold:  dict[int, float] = {}

        self._card_rects:  dict[int, pygame.Rect] = {}
        self._reset_rects: dict[int, pygame.Rect] = {}
        self._start_rects: dict[int, pygame.Rect] = {}
        
        # UI 상태
        self.show_mask = True
        self.video_paused = False
        self.dev_mode = False

        # 영상 오버레이 상태 — 다카메라 그리드용
        # source_id → {"rect": pygame.Rect, "scale": float, "frame_size": (w, h)}
        self._cam_cells: dict[int, dict] = {}
        # 단일 카메라 호환용 (마지막으로 렌더링된 셀)
        self._cam_rect = pygame.Rect(0, 0, 0, 0)
        self._cam_scale = 1.0
        # 캘리브레이션 시 마지막으로 드래그가 시작된 source_id
        self._calib_active_source: int | None = None

        # 캘리브레이션 (설정) 모드
        self.calibration_mode = False
        self._calib_drag_start = None
        self._calib_drag_end = None
        self._calib_dragging = False
        self._calib_burners = []
        self._calib_selected_idx: int | None = None  # 선택된 화구 인덱스

        # 외부 콜백
        self.on_camera_switch = None
        self.on_config_reloaded = None
        self.on_camera_add = None       # 콜백() → 새 source dict 추가 후 reload 트리거 / None 반환 시 실패
        self.on_camera_remove = None    # 콜백(source_id) → 제거 후 reload

        # 카메라 +/- 버튼 hitbox
        self._cam_add_rect: pygame.Rect | None = None
        self._cam_remove_rect: pygame.Rect | None = None
        self._toast_msg: str | None = None
        self._toast_until: float = 0.0

        # 신규 추가 상태 변수 (소리 및 트래킹)
        self.mute = False
        self._sound_rect: pygame.Rect | None = None
        self._sound_warn_30s = None
        self._sound_complete = None
        self._sound_cam_lost = None
        self._prev_remaining: dict[int, float] = {}
        self._prev_states: dict[int, BurnerState] = {}
        self._cam_offline_since: dict[int, float] = {}
        self._cam_lost_played: dict[int, bool] = {}
        self._sorted_burners_cache = []

        # 작업 G — 패널 가시성 조정
        self._right_panel_w: int = self._cfg.get("right_panel_w", _RIGHT_PANEL_W)
        self._video_hidden: bool = False
        self._dragging_divider: bool = False
        # 카메라 풀뷰(단독 확대) — 영상 더블클릭으로 토글
        self._focus_camera_id: int | None = None
        self._last_click_src: int | None = None
        self._last_click_time: float = 0.0

    def init(self) -> None:
        pygame.init()
        # Responsive 기본 1280x720
        w, h = self._cfg.get("window_size", [1280, 720])
        w = max(w, 1024)
        h = max(h, 600)
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption("길가옆에 압력밥솥 타이머 시스템")
        self._clock = pygame.time.Clock()
        self._fonts = _load_fonts()

        # Pygame 오디오 믹서 초기화 (사운드 에셋 로드)
        try:
            pygame.mixer.init()
            sound_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sounds")
            w_path = os.path.join(sound_dir, "warn_30s.wav")
            c_path = os.path.join(sound_dir, "complete.wav")
            l_path = os.path.join(sound_dir, "cam_lost.wav")
            
            # 절대경로 폴백 지원
            if not os.path.exists(w_path):
                w_path = "assets/sounds/warn_30s.wav"
            if not os.path.exists(c_path):
                c_path = "assets/sounds/complete.wav"
            if not os.path.exists(l_path):
                l_path = "assets/sounds/cam_lost.wav"

            if os.path.exists(w_path):
                self._sound_warn_30s = pygame.mixer.Sound(w_path)
            if os.path.exists(c_path):
                self._sound_complete = pygame.mixer.Sound(c_path)
            if os.path.exists(l_path):
                self._sound_cam_lost = pygame.mixer.Sound(l_path)
            print("[UI] pygame.mixer 사운드 로드 성공")
        except Exception as e:
            print(f"[UI] pygame.mixer 초기화 실패 (무음 작동): {e}")

        # Config에 화구가 없으면 자동 캘리브레이션 진입
        if not self.config_data.get("burners", []):
            self.start_calibration()

    def quit(self) -> None:
        pygame.quit()

    def start_calibration(self):
        self.calibration_mode = True
        self._calib_burners = [dict(b) for b in self.config_data.get("burners", [])]
        self._calib_drag_start = None
        self._calib_drag_end = None
        self._calib_dragging = False
        self._calib_selected_idx = None
        self._selected_id = None
        self._selected_camera_id = None  # 캘리브 진입 시 카메라 선택 해제
        print("[UI] 캘리브레이션 가이드 모드 시작")

    def _renumber_calib(self) -> None:
        """저장 정책: source_id 오름차순 + 같은 카메라 안에선 list 위치(=그린 순서) 유지.

        결과 예시:
            Cam-0 화구 2개 → ID 1, 2
            Cam-1 화구 3개 → ID 3, 4, 5
            Cam-2 화구 1개 → ID 6

        선택된 화구 인덱스도 재정렬 후 동일 화구를 가리키도록 갱신.
        """
        selected_obj = (
            self._calib_burners[self._calib_selected_idx]
            if self._calib_selected_idx is not None and self._calib_selected_idx < len(self._calib_burners)
            else None
        )

        # source_id 오름차순 정렬 (안정 정렬이라 같은 source 안의 그린 순서는 보존됨)
        self._calib_burners.sort(key=lambda b: (b.get("source_id", 0)))

        # 1부터 ID 재부여
        for i, b in enumerate(self._calib_burners):
            b["id"] = i + 1

        # 선택 인덱스 재바인딩
        if selected_obj is not None:
            try:
                self._calib_selected_idx = self._calib_burners.index(selected_obj)
            except ValueError:
                self._calib_selected_idx = None

    def _move_selected_within_camera(self, direction: int) -> bool:
        """선택된 화구를 같은 카메라 안에서 한 칸 이동. direction=-1=앞, +1=뒤. 성공 시 True."""
        if self._calib_selected_idx is None:
            return False
        n = len(self._calib_burners)
        if not (0 <= self._calib_selected_idx < n):
            return False
        cur_idx = self._calib_selected_idx
        cur = self._calib_burners[cur_idx]
        sid = cur.get("source_id", 0)

        # 이동 대상 인덱스 — 같은 source_id 안의 인접 항목
        step = 1 if direction > 0 else -1
        target_idx = cur_idx + step
        while 0 <= target_idx < n:
            if self._calib_burners[target_idx].get("source_id", 0) == sid:
                # 두 항목 swap
                self._calib_burners[cur_idx], self._calib_burners[target_idx] = (
                    self._calib_burners[target_idx], self._calib_burners[cur_idx],
                )
                self._calib_selected_idx = target_idx
                self._renumber_calib()
                return True
            # 다른 카메라 만나면 더 못 감
            if self._calib_burners[target_idx].get("source_id", 0) != sid:
                break
            target_idx += step
        return False

    def _show_toast(self, msg: str, duration_s: float = 2.5) -> None:
        self._toast_msg = msg
        self._toast_until = time.monotonic() + duration_s

    def save_calibration(self):
        self.calibration_mode = False
        # 저장 직전 자동 그룹화 — source_id 오름차순, 같은 카메라 내 list 순서 유지, ID 1부터
        self._renumber_calib()
        self.config_data["burners"] = self._calib_burners
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        print(f"[UI] 캘리브레이션 저장 완료 ({len(self._calib_burners)}개 화구)")
        
        # 콜백이 있다면 즉시 반영
        if getattr(self, "on_config_reloaded", None):
            self.on_config_reloaded(self.config_data)
        else:
            print("[UI] 변경된 설정은 다음 실행 혹은 내부 레지스트리 재등록 후 적용됩니다.")

    # ── Pygame Event Handling ────────────────────────────────────────────────
    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return True

        if event.type == pygame.VIDEORESIZE:
            w, h = max(event.w, 900), max(event.h, 600)
            self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            return False

        if event.type == pygame.KEYDOWN:
            return self._on_keydown(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_mouse_down(event.pos)

        if event.type == pygame.MOUSEMOTION:
            self._on_mouse_move(event.pos)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._on_mouse_up(event.pos)

        return False

    def _on_keydown(self, event: pygame.event.Event) -> bool:
        key = event.key
        
        # 공통 단축키
        if key == pygame.K_q:
            return True # 종료
            
        if key == pygame.K_SPACE:
            self.video_paused = not self.video_paused
            return False

        if self.calibration_mode:
            # 캘리브레이션 전용 키
            if key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
                self.save_calibration()
            elif key == pygame.K_ESCAPE:
                self.calibration_mode = False  # 원복
                self._calib_selected_idx = None
            elif key == pygame.K_z:
                if self._calib_burners:
                    self._calib_burners.pop()
                    self._renumber_calib()
                    self._calib_selected_idx = None
            elif key in (pygame.K_DELETE, pygame.K_BACKSPACE):
                if self._calib_selected_idx is not None and 0 <= self._calib_selected_idx < len(self._calib_burners):
                    self._calib_burners.pop(self._calib_selected_idx)
                    self._renumber_calib()
                    self._calib_selected_idx = None
            elif key == pygame.K_LEFTBRACKET:
                # 선택된 화구를 같은 카메라 안에서 한 칸 앞으로 (ID ↓)
                self._move_selected_within_camera(-1)
            elif key == pygame.K_RIGHTBRACKET:
                # 선택된 화구를 같은 카메라 안에서 한 칸 뒤로 (ID ↑)
                self._move_selected_within_camera(+1)
            return False

        # 일반 운용 단축키
        if key == pygame.K_m:
            self.show_mask = not self.show_mask
        elif key == pygame.K_c:
            if self.on_camera_switch:
                self.on_camera_switch(self._selected_camera_id)
        elif key == pygame.K_n:
            self.mute = not self.mute
            self._show_toast("음소거 ON" if self.mute else "음소거 OFF")
        elif key == pygame.K_ESCAPE:
            self._selected_id = None
            self._selected_camera_id = None
            self._focus_camera_id = None
        elif key == pygame.K_f:
            self.start_calibration()
        elif key == pygame.K_d:
            self.dev_mode = not self.dev_mode
        elif key == pygame.K_v:
            self._video_hidden = not self._video_hidden
            self._show_toast("영상 숨김 (카드만 보기)" if self._video_hidden else "영상 표시")

        # 화구 선택
        num_map = {
            pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3,
            pygame.K_4: 4, pygame.K_5: 5, pygame.K_6: 6,
            pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9,
            pygame.K_0: 10,
        }
        if key in num_map:
            bid = num_map[key]
            try:
                self._registry.get(bid)
                self._selected_id = bid
            except KeyError:
                pass

        if self._selected_id is not None:
            bsm = self._registry.get(self._selected_id)
            if key == pygame.K_r:
                bsm.manual_reset()
            elif key == pygame.K_s:
                bsm.manual_start()

        return False

    # ── Mouse Handling ───────────────────────────────────────────────────────
    def _to_video_pos(self, pos: tuple[int, int]) -> tuple[int, int, int] | None:
        """Pygame 좌표 → (source_id, vx, vy). 어느 카메라 셀에도 안 걸리면 None."""
        for src_id, cell in self._cam_cells.items():
            rect = cell["rect"]
            if rect.collidepoint(pos):
                scale = cell["scale"]
                vx = (pos[0] - rect.x) / scale
                vy = (pos[1] - rect.y) / scale
                return (src_id, int(vx), int(vy))
        return None

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.calibration_mode:
            v_pos = self._to_video_pos(pos)
            if v_pos:
                src_id, vx, vy = v_pos
                # 기존 ROI 클릭 확인 (역순 — 위에 그려진 게 우선)
                hit_idx = None
                for idx in range(len(self._calib_burners) - 1, -1, -1):
                    b = self._calib_burners[idx]
                    if b.get("source_id") != src_id:
                        continue
                    rx, ry, rw, rh = b["roi"]
                    if rx <= vx <= rx + rw and ry <= vy <= ry + rh:
                        hit_idx = idx
                        break
                if hit_idx is not None:
                    # 선택만 (드래그 시작 X)
                    self._calib_selected_idx = hit_idx
                    self._calib_dragging = False
                    self._calib_drag_start = None
                    self._calib_drag_end = None
                else:
                    # 새 ROI 드래그 시작
                    self._calib_selected_idx = None
                    self._calib_active_source = src_id
                    self._calib_dragging = True
                    self._calib_drag_start = (vx, vy)
                    self._calib_drag_end = (vx, vy)
            else:
                # 영상 영역 바깥 클릭 → 선택 해제
                self._calib_selected_idx = None
            return

        # 분할선 드래그 시작 (영상 표시 중일 때만, 다른 클릭보다 우선)
        if not self._video_hidden and self._screen is not None:
            divider_x = self._screen.get_width() - self._right_panel_w
            if abs(pos[0] - divider_x) <= 6:
                self._dragging_divider = True
                return

        # 소리 토글 버튼 클릭 (px 영역 정의 필요하나 렌더링 때 저장해 둔 Rect 사용)
        if self._sound_rect and self._sound_rect.collidepoint(pos):
            self.mute = not self.mute
            self._show_toast("음소거 ON" if self.mute else "음소거 OFF")
            return

        # 카메라 +/- 버튼
        if self._cam_add_rect and self._cam_add_rect.collidepoint(pos):
            if self.on_camera_add:
                ok = self.on_camera_add()
                self._show_toast("카메라 추가됨" if ok else "추가 가능한 카메라 없음")
            return
        if self._cam_remove_rect and self._cam_remove_rect.collidepoint(pos):
            if self._selected_camera_id is None:
                self._show_toast("제거할 카메라를 영상에서 먼저 클릭하세요")
            elif self.on_camera_remove:
                target_sid = self._selected_camera_id
                ok = self.on_camera_remove(target_sid)
                if ok:
                    self._show_toast(f"Cam-{target_sid} 제거됨")
                    self._selected_camera_id = None
                else:
                    self._show_toast(f"Cam-{target_sid} 제거 실패")
            return

        # 일반 모드 - 화구 리셋
        for bid, rect in self._reset_rects.items():
            if rect.collidepoint(pos):
                bsm = self._registry.get(bid)
                if bsm.state == BurnerState.DONE_SECOND:
                    bsm.manual_reset()
                else:
                    self._reset_hold[bid] = time.monotonic()
                return

        # 일반 모드 - 화구 수동 시작
        for bid, rect in self._start_rects.items():
            if rect.collidepoint(pos):
                self._registry.get(bid).manual_start()
                return

        # 카드 선택
        for bid, rect in self._card_rects.items():
            if rect.collidepoint(pos):
                self._selected_id = bid if self._selected_id != bid else None
                return

        # 영상 영역 클릭 → 단일클릭=카메라 선택 토글 / 더블클릭=풀뷰 토글
        v_pos = self._to_video_pos(pos)
        if v_pos:
            src_id = v_pos[0]
            now = time.monotonic()
            if self._last_click_src == src_id and (now - self._last_click_time) < 0.4:
                # 더블클릭 → 풀뷰(단독 확대) 토글
                self._focus_camera_id = None if self._focus_camera_id == src_id else src_id
                self._show_toast("풀뷰 (더블클릭으로 그리드 복귀)" if self._focus_camera_id else "그리드 보기")
                self._last_click_src = None
            else:
                self._selected_camera_id = src_id if self._selected_camera_id != src_id else None
                self._last_click_src = src_id
                self._last_click_time = now
            return

    def _on_mouse_move(self, pos: tuple[int, int]) -> None:
        if self._dragging_divider and self._screen is not None:
            sw = self._screen.get_width()
            # 패널 폭 = 창 너비 - 마우스 x. MIN=300(카드 최소), MAX=sw(영상 0 = 카드 전체)
            self._right_panel_w = max(300, min(sw, sw - pos[0]))
            return
        if self.calibration_mode and self._calib_dragging:
            v_pos = self._to_video_pos(pos)
            if v_pos and v_pos[0] == self._calib_active_source:
                # 드래그가 다른 카메라 셀로 넘어가는 건 무시 (시작 셀 안에서만)
                self._calib_drag_end = (v_pos[1], v_pos[2])

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self._dragging_divider:
            self._dragging_divider = False
            # 변경된 폭을 ui 설정에 기록(다음 캘리브 저장 시 함께 영구 저장됨)
            self._cfg["right_panel_w"] = self._right_panel_w
            return
        if self.calibration_mode and self._calib_dragging:
            self._calib_dragging = False
            if self._calib_drag_start and self._calib_drag_end and self._calib_active_source is not None:
                x1, y1 = self._calib_drag_start
                x2, y2 = self._calib_drag_end
                roi = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                if roi[2] > 20 and roi[3] > 20:
                    next_id = len(self._calib_burners) + 1
                    self._calib_burners.append({
                        "id": next_id,
                        "source_id": self._calib_active_source,
                        "countdown_first": 720,
                        "countdown_second": 300,
                        "done_first_timeout": 600,
                        "pot_absent_threshold": 30,
                        "roi": roi
                    })
            self._calib_drag_start = None
            self._calib_drag_end = None
            self._calib_active_source = None
            return

        self._reset_hold.clear()

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self, frames: dict = None, processor = None) -> None:
        if self._screen is None:
            return

        # 길게 누르기 버튼 처리
        now = time.monotonic()
        for bid, start_t in list(self._reset_hold.items()):
            if now - start_t >= _RESET_HOLD_S:
                self._registry.get(bid).manual_reset()
                del self._reset_hold[bid]

        sw, sh = self._screen.get_size()
        self._screen.fill(_C_BG)

        # 패널 폭 결정 — 영상 숨김 시 패널이 전체 폭. 창 크기에 맞춰 클램프.
        if self._video_hidden:
            panel_w = sw
        else:
            panel_w = max(300, min(sw, self._right_panel_w))
            self._right_panel_w = panel_w
        main_w = sw - panel_w

        # 1. 왼쪽 카메라 그리드 렌더링 (영상 영역이 있을 때만)
        if main_w > 0:
            self._draw_camera_grid(frames or {}, processor, main_w, sh)
            if self.calibration_mode:
                self._draw_calibration_overlay(main_w, sh)

        # 2. 오른쪽 제어 패널 렌더링
        right_panel = pygame.Rect(main_w, 0, panel_w, sh)
        pygame.draw.rect(self._screen, _C_PANEL, right_panel)
        # 패널 왼쪽 구분선 + 드래그 핸들 (영상 표시 중일 때만)
        if main_w > 0:
            pygame.draw.line(self._screen, _C_CARD_BORDER, (main_w, 0), (main_w, sh), 2)
            self._draw_divider_handle(main_w, sh)

        self._draw_right_panel(main_w, sh, panel_w)

        # Toast (있으면)
        if self._toast_msg and time.monotonic() < self._toast_until:
            toast = self._fonts["label"].render(self._toast_msg, True, _C_TEXT_DARK)
            pad = 14
            box = pygame.Rect(0, 0, toast.get_width() + pad * 2, toast.get_height() + 12)
            box.midbottom = (main_w // 2, sh - 24)
            pygame.draw.rect(self._screen, _C_BRAND, box, border_radius=6)
            self._screen.blit(toast, toast.get_rect(center=box.center))

        pygame.display.flip()
        if self._clock:
            self._clock.tick(30) # 30fps UI Redraw

    def _draw_divider_handle(self, x: int, sh: int) -> None:
        """분할선 중앙에 세로 그립(점 3개) — 끌 수 있다는 발견성 표시."""
        cy = sh // 2
        for dy in (-12, 0, 12):
            pygame.draw.circle(self._screen, _C_BRAND, (x, cy + dy), 2)

    def _grid_dims(self, n: int) -> tuple[int, int]:
        """카메라 N대 → (cols, rows). 1:풀화면, 2:1×2, 3~4:2×2, 5~6:3×2, 7~9:3×3, …"""
        if n <= 1: return (1, 1)
        if n == 2: return (2, 1)
        if n <= 4: return (2, 2)
        if n <= 6: return (3, 2)
        if n <= 9: return (3, 3)
        if n <= 12: return (4, 3)
        return (4, 4)

    def _draw_camera_grid(self, frames: dict, processor, box_w: int, box_h: int):
        """카메라 N대를 균등 그리드로 표시. 각 셀의 _cam_cells 기록.

        풀뷰(_focus_camera_id) 지정 시 해당 카메라만 영상 영역 전체로 크게 표시.
        오프라인 추적은 풀뷰 여부와 무관하게 전체 카메라 대상으로 수행.
        """
        all_sources = self.config_data.get("sources", [])
        if not all_sources:
            return

        # 오프라인 상태 지속 시간 체크 및 소리 재생 (전체 카메라 대상)
        now = time.monotonic()
        for sc in all_sources:
            sid = sc["id"]
            frame = frames.get(sid)
            if frame is None:
                if sid not in self._cam_offline_since:
                    self._cam_offline_since[sid] = now
            else:
                self._cam_offline_since.pop(sid, None)
                self._cam_lost_played.pop(sid, None)

        # 렌더 대상 — 풀뷰면 포커스 카메라만 (캘리브 모드에서는 항상 전체)
        sources_cfg = all_sources
        if self._focus_camera_id is not None and not self.calibration_mode:
            focused = [s for s in all_sources if s["id"] == self._focus_camera_id]
            if focused:
                sources_cfg = focused
            else:
                self._focus_camera_id = None  # 포커스 카메라가 사라졌으면 해제

        n = len(sources_cfg)
        cols, rows = self._grid_dims(n)
        cell_w = (box_w - _CELL_GAP * (cols + 1)) // cols
        cell_h = (box_h - _CELL_GAP * (rows + 1)) // rows

        self._cam_cells.clear()

        for i, sc in enumerate(sources_cfg):
            src_id = sc["id"]
            r = i // cols
            c = i % cols
            ox = _CELL_GAP + c * (cell_w + _CELL_GAP)
            oy = _CELL_GAP + r * (cell_h + _CELL_GAP)

            cell_bg = pygame.Rect(ox, oy, cell_w, cell_h)
            pygame.draw.rect(self._screen, _C_PANEL, cell_bg, border_radius=4)

            frame = frames.get(src_id)
            self._draw_camera_cell(sc, frame, processor, ox, oy, cell_w, cell_h)

            # 카메라 유실 3초 이상이면 빨간색 점멸 테두리
            is_lost = False
            if src_id in self._cam_offline_since:
                if now - self._cam_offline_since[src_id] >= 3.0:
                    is_lost = True
                    # 최초 경보음 재생
                    if not self.mute and self._sound_cam_lost:
                        if not self._cam_lost_played.get(src_id, False):
                            self._sound_cam_lost.play()
                            self._cam_lost_played[src_id] = True

            if is_lost:
                if int(time.time() * 3) % 2 == 0:
                    pygame.draw.rect(self._screen, _C_FAILED, cell_bg, width=4, border_radius=4)
            # 선택된 카메라면 외곽에 노란 스트로크
            elif self._selected_camera_id == src_id:
                pygame.draw.rect(self._screen, _C_BRAND, cell_bg, width=4, border_radius=4)

    def _draw_camera_cell(self, sc: dict, frame, processor, ox: int, oy: int, cw: int, ch: int):
        """단일 카메라 셀 렌더링. frame이 None이면 '연결 안 됨' 표시."""
        src_id = sc["id"]
        label = sc.get("label", f"Cam-{src_id}")

        # 헤더 (라벨 + 연결 상태) — 가독성 위해 크게
        header_h = 30
        header_rect = pygame.Rect(ox, oy, cw, header_h)
        pygame.draw.rect(self._screen, _C_CARD_BG, header_rect)
        pygame.draw.line(self._screen, _C_BRAND, (ox, oy + header_h - 1), (ox + cw, oy + header_h - 1), 1)

        if frame is None:
            status_color = _C_FAILED
            status_text = "● 프레임 없음"
        else:
            status_color = _C_CONNECTED
            status_text = "● 연결됨"

        pygame.draw.circle(self._screen, status_color, (ox + 12, oy + header_h // 2), 6)
        label_surf = self._fonts["label"].render(f"[Cam-{src_id}]  {label}", True, _C_BRAND)
        self._screen.blit(label_surf, (ox + 26, oy + 6))
        status_surf = self._fonts["small"].render(status_text, True, status_color)
        self._screen.blit(status_surf, (ox + cw - status_surf.get_width() - 8, oy + 8))

        video_area_y = oy + header_h
        video_area_h = ch - header_h

        if frame is None:
            no_signal = self._fonts["label"].render("⛔ 신호 없음", True, _C_FAILED)
            self._screen.blit(no_signal, no_signal.get_rect(center=(ox + cw // 2, video_area_y + video_area_h // 2)))
            # 좌표 매핑은 비워둠 (드래그 불가)
            return

        # ── 이 카메라 소속 화구의 ROI / 감지 결과만 오버레이 ─────────────
        vis = frame.copy()

        if not self.calibration_mode:
            burners_of_src = [b for b in self.config_data.get("burners", []) if b.get("source_id") == src_id]
            for cfg in burners_of_src:
                bid = cfg["id"]
                try:
                    bsm = self._registry.get(bid)
                except KeyError:
                    continue

                if "roi" in cfg:
                    x, y, w, h = cfg["roi"]
                    cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

                if processor:
                    r_c, g_c, b_c = bsm.color
                    color_bgr = (b_c, g_c, r_c)

                    if bid in processor.last_matched_boxes:
                        bx1, by1, bx2, by2 = processor.last_matched_boxes[bid]
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color_bgr, 1)
                        length = 12
                        for pt1, pt2 in [
                            ((bx1, by1), (bx1+length, by1)), ((bx1, by1), (bx1, by1+length)),
                            ((bx2, by1), (bx2-length, by1)), ((bx2, by1), (bx2, by1+length)),
                            ((bx1, by2), (bx1+length, by2)), ((bx1, by2), (bx1, by2-length)),
                            ((bx2, by2), (bx2-length, by2)), ((bx2, by2), (bx2, by2-length))
                        ]:
                            cv2.line(vis, pt1, pt2, color_bgr, 2)

                        text = f"#{bid}"
                        fs = 1.2 if self.dev_mode else 0.5
                        thick = 3 if self.dev_mode else 1
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
                        cv2.rectangle(vis, (bx1, by1 - th - 6), (bx1 + tw + 6, by1), color_bgr, -1)
                        text_c = (0, 0, 0) if (g_c > 150 or r_c > 150) else (255, 255, 255)
                        cv2.putText(vis, text, (bx1 + 3, by1 - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, text_c, thick)

                    if bid in processor.last_weight_boxes:
                        wx1, wy1, wx2, wy2 = processor.last_weight_boxes[bid]
                        if self.show_mask and bid in processor.last_mask_xys:
                            pts = processor.last_mask_xys[bid].astype(np.int32)
                            overlay = vis.copy()
                            cv2.fillPoly(overlay, [pts], (0, 200, 255))
                            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                            cv2.polylines(vis, [pts], True, (0, 180, 255), 1)
                        if self.show_mask:
                            cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 255, 255), 1)
                            score = bsm.vibration_score
                            cv2.rectangle(vis, (wx1, wy2+3), (wx2, wy2+8), (60, 60, 60), -1)
                            fill_w = int((wx2 - wx1) * min(score, 1.0))
                            cgauge = (0, 200, 0) if score < 1.0 else (0, 60, 255)
                            if fill_w > 0:
                                cv2.rectangle(vis, (wx1, wy2+3), (wx1+fill_w, wy2+8), cgauge, -1)
                            if self.dev_mode and bsm.current_angle is not None:
                                score_txt = f"RMS: {bsm.current_angle:.3f}"
                                cv2.putText(vis, score_txt, (wx1, wy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # BGR → RGB 변환 + 비디오 영역에 맞춰 스케일
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        scale = min(cw / ow, video_area_h / oh)
        nw, nh = max(1, int(ow * scale)), max(1, int(oh * scale))
        rgb = cv2.resize(rgb, (nw, nh))
        surf = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), 'RGB')

        cx = ox + (cw - nw) // 2
        cy = video_area_y + (video_area_h - nh) // 2
        cell_rect = pygame.Rect(cx, cy, nw, nh)
        self._screen.blit(surf, (cx, cy))

        # 셀 좌표 매핑 저장 (마우스 이벤트에서 사용)
        self._cam_cells[src_id] = {
            "rect": cell_rect,
            "scale": scale,
            "frame_size": (ow, oh),
        }
        # 단일 카메라 호환용
        self._cam_rect = cell_rect
        self._cam_scale = scale

        if self.video_paused:
            pause_surf = pygame.Surface((nw, nh), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 120))
            txt = self._fonts["small"].render("일시정지", True, _C_BRAND)
            pause_surf.blit(txt, txt.get_rect(center=(nw//2, nh//2)))
            self._screen.blit(pause_surf, (cx, cy))

    def _draw_calibration_overlay(self, box_w, box_h):
        """캘리브레이션 모드 — 확정된 화구를 각 source_id 셀에 매핑해서 표시."""
        # 1. 확정된 화구들 (셀별 좌표 변환)
        for idx, b in enumerate(self._calib_burners):
            src_id = b.get("source_id", 0)
            cell = self._cam_cells.get(src_id)
            if cell is None:
                continue
            rx, ry, rw, rh = b["roi"]
            scale = cell["scale"]
            sx = int(rx * scale) + cell["rect"].x
            sy = int(ry * scale) + cell["rect"].y
            sw = int(rw * scale)
            sh = int(rh * scale)

            is_selected = (idx == self._calib_selected_idx)
            border_color = (255, 80, 80) if is_selected else _C_BRAND
            border_w = 4 if is_selected else 2
            pygame.draw.rect(self._screen, border_color, (sx, sy, sw, sh), border_w)
            pygame.draw.rect(self._screen, border_color, (sx, sy, 26, 22))
            txt = self._fonts["id"].render(str(b["id"]), True, _C_TEXT_DARK)
            self._screen.blit(txt, (sx + 4, sy + 1))

        # 2. 드래그 중인 임시 사각형 (활성 셀 위에)
        if self._calib_dragging and self._calib_drag_start and self._calib_drag_end and self._calib_active_source is not None:
            cell = self._cam_cells.get(self._calib_active_source)
            if cell:
                rx = min(self._calib_drag_start[0], self._calib_drag_end[0])
                ry = min(self._calib_drag_start[1], self._calib_drag_end[1])
                rw = abs(self._calib_drag_end[0] - self._calib_drag_start[0])
                rh = abs(self._calib_drag_end[1] - self._calib_drag_start[1])
                scale = cell["scale"]
                sx = int(rx * scale) + cell["rect"].x
                sy = int(ry * scale) + cell["rect"].y
                sw = int(rw * scale)
                sh = int(rh * scale)
                pygame.draw.rect(self._screen, (255, 100, 100), (sx, sy, sw, sh), 2)

        # 3. 상단 안내 바
        banner = pygame.Surface((box_w, 60), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 180))
        self._screen.blit(banner, (0, 0))
        title = self._fonts["title"].render("🛠 화구 설정 (Calibration) 모드", True, _C_BRAND)
        n_per_src: dict[int, int] = {}
        for b in self._calib_burners:
            sid = b.get("source_id", 0)
            n_per_src[sid] = n_per_src.get(sid, 0) + 1
        summary = "  ".join(f"Cam-{sid}:{n}" for sid, n in sorted(n_per_src.items())) or "아직 화구 없음"
        sel_hint = f" | 선택:#{self._calib_burners[self._calib_selected_idx]['id']} (DEL=삭제, [ ]=순서변경)" \
                   if self._calib_selected_idx is not None and self._calib_selected_idx < len(self._calib_burners) else ""
        desc = self._fonts["small"].render(
            f"드래그=새화구  클릭=선택  DEL=삭제  [ ]=ID앞뒤  Z=이전취소  ENTER=저장  ESC=취소  ({summary}){sel_hint}",
            True, _C_TEXT_LIGHT
        )
        self._screen.blit(title, (20, 8))
        self._screen.blit(desc, (20, 36))

    def _draw_right_panel(self, px: int, ph: int, panel_w: int = _RIGHT_PANEL_W):
        # 상단 헤더 (로고/타이틀 영역)
        header_h = 70
        pygame.draw.rect(self._screen, _C_BRAND, (px, 0, panel_w, header_h))
        title_surf = self._fonts["title"].render("길가옆에 압력밥솥 타이머", True, _C_TEXT_DARK)
        self._screen.blit(title_surf, (px + 20, 20))
        
        # 모델 부재 경고
        oy = header_h + 10
        if self._model_missing:
            wrng = pygame.Rect(px + 10, oy, panel_w - 20, 36)
            pygame.draw.rect(self._screen, _C_WARNING_BG, wrng, border_radius=6)
            msg = self._fonts["small"].render("⚠ AI 모델 없음 (수동 구동만 가능)", True, _C_TEXT_LIGHT)
            self._screen.blit(msg, (px + 20, oy + 10))
            oy += 46

        # 상태 및 툴바
        toolbar_rect = pygame.Rect(px + 10, oy, panel_w - 20, 64)
        pygame.draw.rect(self._screen, _C_CARD_BG, toolbar_rect, border_radius=6)
        
        conf_btn = self._fonts["small"].render("⚙ 설정(F)", True, _C_BRAND if not self.calibration_mode else _C_TEXT_LIGHT)
        mask_btn = self._fonts["small"].render(f"Mask(M): {'ON' if self.show_mask else 'OFF'}", True, _C_BRAND if self.show_mask else _C_TEXT_LIGHT)
        cam_btn = self._fonts["small"].render("카메라 전환(C)", True, _C_TEXT_LIGHT)
        dev_btn = self._fonts["small"].render(f"Dev(D): {'ON' if getattr(self, 'dev_mode', False) else 'OFF'}", True, _C_BRAND if getattr(self, 'dev_mode', False) else _C_TEXT_LIGHT)
        play_btn = self._fonts["small"].render("일시정지(Space)" if not self.video_paused else "재생(Space)", True, _C_BRAND if self.video_paused else _C_TEXT_LIGHT)

        self._screen.blit(conf_btn, (px + 20, oy + 12))
        self._screen.blit(mask_btn, (px + 110, oy + 12))
        self._screen.blit(cam_btn, (px + 210, oy + 12))
        
        self._screen.blit(dev_btn, (px + 20, oy + 36))
        self._screen.blit(play_btn, (px + 110, oy + 36))

        # 소리 ON/OFF 버튼 렌더링 및 클릭 영역 지정
        sound_btn = self._fonts["small"].render(f"소리(N): {'OFF' if self.mute else 'ON'}", True, _C_BRAND if not self.mute else _C_FAILED)
        self._screen.blit(sound_btn, (px + 210, oy + 36))
        self._sound_rect = pygame.Rect(px + 210, oy + 32, 100, 24)

        oy += 74

        # 카메라 관리 — 현재 대수 + +/- 버튼
        n_cams = len(self.config_data.get("sources", []))
        cam_bar = pygame.Rect(px + 10, oy, panel_w - 20, 44)
        pygame.draw.rect(self._screen, _C_CARD_BG, cam_bar, border_radius=6)

        sel_hint = f"  (선택: Cam-{self._selected_camera_id})" if self._selected_camera_id is not None else ""
        cam_label = self._fonts["small"].render(f"카메라 {n_cams}대{sel_hint}", True, _C_TEXT_LIGHT)
        self._screen.blit(cam_label, (px + 20, oy + 14))

        btn_w, btn_h = 70, 28
        add_x = px + panel_w - 20 - btn_w
        rem_x = add_x - btn_w - 6
        self._cam_remove_rect = pygame.Rect(rem_x, oy + 8, btn_w, btn_h)
        self._cam_add_rect = pygame.Rect(add_x, oy + 8, btn_w, btn_h)

        # 제거 버튼: 카메라 선택 시 강조, 미선택 시 회색
        rem_active = self._selected_camera_id is not None
        rem_color = (180, 60, 60) if rem_active else (60, 60, 70)
        rem_text = f"− Cam-{self._selected_camera_id}" if rem_active else "− 제거"
        pygame.draw.rect(self._screen, rem_color, self._cam_remove_rect, border_radius=4)
        pygame.draw.rect(self._screen, _C_SUCCESS, self._cam_add_rect, border_radius=4)
        rem_surf = self._fonts["btn"].render(rem_text, True, _C_TEXT_LIGHT)
        add_surf = self._fonts["btn"].render("+ 추가", True, _C_TEXT_LIGHT)
        self._screen.blit(rem_surf, rem_surf.get_rect(center=self._cam_remove_rect.center))
        self._screen.blit(add_surf, add_surf.get_rect(center=self._cam_add_rect.center))

        oy += 54

        if self.calibration_mode:
            # 설정 모드일 때는 카드 리스트 대신 설정 가이드 노출
            guide_y = oy + 40
            c_surf = self._fonts["label"].render("영상 영역에 마우스를 드래그하여", True, _C_TEXT_LIGHT)
            self._screen.blit(c_surf, (px + 40, guide_y))
            c_surf2 = self._fonts["label"].render("불이 나오는 화구 위치를 잡아주세요.", True, _C_TEXT_LIGHT)
            self._screen.blit(c_surf2, (px + 40, guide_y + 30))
            
            c_help = self._fonts["small"].render("단축키안내", True, _C_BRAND)
            self._screen.blit(c_help, (px + 40, guide_y + 100))
            self._screen.blit(self._fonts["small"].render("✔ ENTER : 저장 후 적용", True, _C_TEXT_LIGHT), (px + 40, guide_y + 130))
            self._screen.blit(self._fonts["small"].render("✔ ESC : 저장하지 않고 나가기", True, _C_TEXT_LIGHT), (px + 40, guide_y + 155))
            self._screen.blit(self._fonts["small"].render("✔ Z : 마지막 그린 화구 취소", True, _C_TEXT_LIGHT), (px + 40, guide_y + 180))
            return

        # 스크롤 가능한/리스트 카드 영역 (4열 그리드)
        # 마우스 Hover 판정을 통한 정렬 Freeze 잠금 장치
        mx, my = pygame.mouse.get_pos()
        panel_rect = pygame.Rect(px + 5, oy, panel_w - 10, ph - oy)
        is_hovering = panel_rect.collidepoint((mx, my))

        # 캐시가 없거나 호버 중이 아닐 때만 정렬 순서 업데이트
        if not getattr(self, "_sorted_burners_cache", None) or not is_hovering:
            HOLD_S = 8.0
            def sort_key(b):
                if b.is_counting:
                    return (0, b.remaining_seconds, b.burner_id)
                sd = b.seconds_since_done
                if 0 <= sd <= HOLD_S:
                    return (1, sd, b.burner_id)
                return (2, b.burner_id, 0)
            self._sorted_burners_cache = sorted(self._registry.all(), key=sort_key)

        burners = self._sorted_burners_cache
        if not burners:
            return

        cols = 4
        gap  = 6
        card_w = (panel_w - 10 - (cols - 1) * gap) // cols
        card_h = 102

        self._card_rects.clear()
        self._start_rects.clear()
        self._reset_rects.clear()

        for i, bsm in enumerate(burners):
            r = i // cols
            c = i % cols
            cx = px + 5 + c * (card_w + gap)
            cy = oy + r * (card_h + gap)
            self._draw_burner_card(bsm, cx, cy, card_w, card_h)

    @staticmethod
    def _card_text_color(bg_color: tuple) -> tuple:
        r, g, b = bg_color
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return _C_TEXT_DARK if lum > 0.4 else _C_TEXT_LIGHT

    def _draw_burner_card(self, bsm, x, y, w, h):
        bid = bsm.burner_id
        selected = (bid == self._selected_id)

        # 알림음 재생 제어 (작업 E)
        cur_rem = bsm.remaining_seconds
        prev_rem = self._prev_remaining.get(bid, cur_rem)

        # 1) 완료 30초 전 예고 (남은 시간이 30초 선을 막 통과했을 때)
        from core.state_machine import BurnerState as BS
        if bsm.is_counting and prev_rem > 30 >= cur_rem > 0:
            if not self.mute and self._sound_warn_30s:
                self._sound_warn_30s.play()

        # 2) 완료 순간 전이 (STEAMING -> DONE_FIRST 또는 DONE_SECOND)
        prev_st = self._prev_states.get(bid, bsm.state)
        if prev_st in (BS.POT_STEAMING_FIRST, BS.POT_STEAMING_SECOND) and bsm.state in (BS.DONE_FIRST, BS.DONE_SECOND):
            if not self.mute and self._sound_complete:
                self._sound_complete.play()

        self._prev_remaining[bid] = cur_rem
        self._prev_states[bid] = bsm.state

        # --- 섀도우(그림자) 효과 렌더링 (세련된 입체감) ---
        shadow_rect = pygame.Rect(x + 2, y + 3, w - 2, h - 2)
        pygame.draw.rect(self._screen, (15, 15, 18), shadow_rect, border_radius=8)

        card_rect = pygame.Rect(x, y, w - 2, h - 2)
        self._card_rects[bid] = card_rect

        # 고급스럽게 조율된 세련된 다크 테마 색상 맵핑
        CUSTOM_COLORS = {
            BS.EMPTY:               (42,  42,  48),   # 차분한 다크 그레이
            BS.POT_IDLE:            (225, 165,  10),   # 골드 머스터드
            BS.POT_STEAMING_FIRST:  (40,  167,  69),   # 부트스트랩 그린
            BS.DONE_FIRST:          (253, 126,  20),   # 세련된 오렌지
            BS.WAIT_SECOND:         (52,  144, 222),   # 소프트 클래식 블루
            BS.POT_STEAMING_SECOND: (25,  110,  25),   # 포레스트 다크 그린
            BS.DONE_SECOND:         (220,  53,  69),   # 팝레드
        }

        # 완료 화구 전역 동기화 점멸 적용
        bg_color = CUSTOM_COLORS.get(bsm.state, (45, 45, 52))
        if bsm.state == BS.DONE_SECOND:
            bg_color = CUSTOM_COLORS[BS.DONE_SECOND] if int(time.time() * 2) % 2 == 0 else (80, 15, 20)

        pygame.draw.rect(self._screen, bg_color, card_rect, border_radius=8)

        # 세련된 보더라인 (선택 테두리는 브랜드 컬러, 평소에는 은은한 반투명 느낌의 테두리)
        bcolor = _C_SELECTED if selected else (80, 80, 90)
        border_w = 2 if selected else 1
        pygame.draw.rect(self._screen, bcolor, card_rect, border_w, border_radius=8)

        tc = self._card_text_color(bg_color)

        # --- 1. 화구 번호 (일반 모드: 좌상단에 크고 굵은 숫자 / dev_mode: 상단 콤팩트 축소) ---
        if self.dev_mode:
            # 개발 모드: 콤팩트한 번호 배지
            rad = 8
            cx, cy = x + 10, y + 11
            pygame.draw.circle(self._screen, _C_BRAND, (cx, cy), rad)
            id_surf = self._fonts["small"].render(str(bid), True, _C_TEXT_DARK)
            self._screen.blit(id_surf, id_surf.get_rect(center=(cx, cy)))
        else:
            # 일반 모드: 크고 수려한 화구 번호 텍스트
            num_str = f"{bid:02d}" if bid >= 10 else str(bid)
            id_surf = self._fonts["card_num"].render(num_str, True, tc)
            self._screen.blit(id_surf, (x + 8, y + 4))

        # --- 2. [Cam-N] 뱃지 (우상단 정렬) ---
        cfg = next((b for b in self.config_data.get("burners", []) if b["id"] == bid), None)
        src_id = cfg.get("source_id", 0) if cfg else 0
        cam_surf = self._fonts["small"].render(f"C{src_id}", True, tc)
        bp = 4
        badge_rect = pygame.Rect(x + w - cam_surf.get_width() - bp * 2 - 8, y + 5,
                                 cam_surf.get_width() + bp * 2, cam_surf.get_height() + 2)
        pygame.draw.rect(self._screen, (30, 30, 35, 100), badge_rect, border_radius=3)
        # 카메라 뱃지 얇은 테두리
        pygame.draw.rect(self._screen, (100, 100, 110, 100), badge_rect, 1, border_radius=3)
        self._screen.blit(cam_surf, (badge_rect.x + bp, badge_rect.y + 1))

        # --- 3. 상태 메시지 및 타이머 (3단 텍스트 중복/침범 소거 완료) ---
        # EMPTY / POT_IDLE / WAIT_SECOND / DONE_SECOND -> 단 1개의 큼직한 한글 텍스트만 노출
        # STEAMING / DONE_FIRST -> 단계 뱃지 + 중앙 큼직한 타이머 노출
        if bsm.state in (BS.EMPTY, BS.POT_IDLE, BS.WAIT_SECOND, BS.DONE_SECOND):
            status_text = {
                BS.EMPTY: "대기",
                BS.POT_IDLE: "준비",
                BS.WAIT_SECOND: "재벌대기",
                BS.DONE_SECOND: "완료!",
            }.get(bsm.state, "대기")
            
            # 중앙에 정갈하게 배치
            st_surf = self._fonts["card_status"].render(status_text, True, tc)
            self._screen.blit(st_surf, (x + 8, y + 34))
        else:
            # 타이머 작동 중인 상태 (STEAMING_FIRST, STEAMING_SECOND, DONE_FIRST)
            # 단계 뱃지 렌더링
            ph_label = bsm.phase_label
            if bsm.state == BS.DONE_FIRST:
                ph_label = "냉각 중"
            
            ph_surf = self._fonts["small"].render(ph_label, True, tc)
            # 단계 뱃지 위치 (화구 번호 우측 공간)
            offset_x = x + 34 if not self.dev_mode else x + 24
            self._screen.blit(ph_surf, (offset_x, y + 8))

            # 중앙 타이머 표시 (가장 큼직하고 뚜렷하게)
            t_color = tc
            if bsm.state == BS.DONE_FIRST:
                t_color = _C_BRAND
            
            # 개발 모드일 때는 겹치지 않게 타이머 크기 조절
            f_timer = self._fonts["label"] if self.dev_mode else self._fonts["card_timer"]
            time_surf = f_timer.render(bsm.status_label, True, t_color)
            self._screen.blit(time_surf, (x + 8, y + 32))

        # --- 4. 개발자 모드 전용 RMS 수치 오버레이 (일반 모드는 완전 격리) ---
        if self.dev_mode:
            rms_val = bsm.current_angle
            rms_text = f"RMS: {rms_val:.3f}" if rms_val is not None else "RMS: -"
            rms_surf = self._fonts["small"].render(rms_text, True, tc)
            self._screen.blit(rms_surf, (x + 8, y + 49))

        # --- 5. 진동 게이지 (슬림하고 예쁜 라운드 게이지바) ---
        gauge_x, gauge_y = x + 8, y + 68
        gauge_w, gauge_h = w - 18, 4
        pygame.draw.rect(self._screen, (30, 30, 35), (gauge_x, gauge_y, gauge_w, gauge_h), border_radius=2)
        score = max(0.0, min(1.0, bsm.vibration_score))
        if score > 0:
            fill_color = _C_SUCCESS if score < 1.0 else _C_WARNING_BG
            pygame.draw.rect(self._screen, fill_color,
                             (gauge_x, gauge_y, int(gauge_w * score), gauge_h), border_radius=2)

        # --- 6. 하단 버튼 영역 (버튼 크기 리스케일로 침범 제거) ---
        if bsm.state in (BS.EMPTY, BS.POT_IDLE): start_label = "시작"
        elif bsm.state == BS.POT_STEAMING_FIRST: start_label = "완료"
        elif bsm.state == BS.DONE_FIRST: start_label = "바로 재벌"
        elif bsm.state == BS.WAIT_SECOND: start_label = "재벌"
        elif bsm.state == BS.POT_STEAMING_SECOND: start_label = "완료"
        else: start_label = "시작"

        bw = (w - 18) // 2
        bh = 20
        by = y + h - bh - 6

        btn_r = pygame.Rect(x + 8, by, bw, bh)
        btn_s = pygame.Rect(x + 8 + bw + 4, by, bw, bh)

        self._reset_rects[bid] = btn_r
        self._start_rects[bid] = btn_s

        hold_prog = min(1.0, (time.monotonic() - self._reset_hold[bid]) / _RESET_HOLD_S) if bid in self._reset_hold else 0.0

        # 버튼 색상 조화
        btn_s_color = _C_SUCCESS
        if bsm.state == BS.DONE_FIRST:
            btn_s_color = (253, 126, 20)  # 세련된 오렌지

        self._draw_btn(btn_r, "R", (70, 70, 75), hold_prog)
        self._draw_btn(btn_s, f"{start_label}(S)" if selected else start_label, btn_s_color)

    def _draw_btn(self, rect: pygame.Rect, text: str, color, hold=0.0):
        pygame.draw.rect(self._screen, color, rect, border_radius=4)
        if hold > 0:
            fill = pygame.Rect(rect.x, rect.y, int(rect.w * hold), rect.h)
            pygame.draw.rect(self._screen, _C_WARNING_BG, fill, border_radius=4)
        
        surf = self._fonts["btn"].render(text, True, _C_TEXT_LIGHT)
        self._screen.blit(surf, surf.get_rect(center=rect.center))

def _load_fonts() -> dict:
    import sys
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/NanumGothic.ttf"
    ]
    font_path = next((p for p in candidates if os.path.exists(p)), None)
    def mk(size, bold=False):
        f = pygame.font.Font(font_path, size) if font_path else pygame.font.Font(None, size + 4)
        if bold:
            f.set_bold(True)
        return f
    return {
        "title": mk(22, bold=True),
        "id": mk(18),
        "label": mk(16),
        "timer": mk(26, bold=True),
        "btn": mk(13),
        "small": mk(13),
        "card_num": mk(20, bold=True),
        "card_timer": mk(22, bold=True),
        "card_status": mk(18, bold=True)
    }
