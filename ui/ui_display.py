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
_RESET_HOLD_S  = 1.0
_CELL_GAP      = 4               # 그리드 셀 간 여백
_HALF_CELL_ASPECT = 0.7           # half_cell_h = half_cell_w * 이 값
_DEFAULT_CAM_SIZE = [8, 7]        # 카메라 박스 기본 크기 (half-cell rows, cols)

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

        # 통합 캔버스 배치 모드 — 카메라 박스 + 화구 카드를 반칸(half-cell) 단위 그리드에 자유 배치
        self.layout_mode = False
        self._layout_cols: int = int(self._cfg.get("layout_cols", 16))  # 반칸 단위 전체 열 수
        self._layout_positions: dict[int, list[int]] = {}         # bid → [row, col] (화구 작업본)
        self._layout_camera_positions: dict[int, list[int]] = {}  # source_id → [row, col] (카메라 작업본)
        self._layout_camera_sizes: dict[int, list[int]] = {}      # source_id → [rows, cols] (카메라 크기 작업본)
        self._layout_drag_kind: str | None = None   # 'burner' | 'camera' | 'resize'
        self._layout_drag_id: int | None = None
        self._layout_drag_pos: tuple[int, int] = (0, 0)
        self._layout_drag_offset: tuple[int, int] = (0, 0)  # 클릭 지점 - 오브젝트 좌측상단 (그랩 오프셋)
        self._layout_btn_rect: pygame.Rect | None = None
        self._cam_box_rects: dict[int, pygame.Rect] = {}       # source_id → 카메라 박스 전체 rect (드래그용)
        self._cam_resize_handles: dict[int, pygame.Rect] = {}  # source_id → 리사이즈 핸들 rect (배치모드)

        # 캔버스(영상+카드 통합 영역) 좌표계 — 매 렌더마다 갱신
        self._canvas_origin: tuple[int, int] = (0, 0)
        self._half_cell_w: int = 0
        self._half_cell_h: int = 0
        self._layout_rows: int = 1  # 보이는 캔버스 영역의 반칸 행 수 (매 프레임 갱신)

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
        self._sound_numbers: dict[int, object] = {}     # 화구 번호(1~30) 안내음
        self._announce_channel = None                    # 번호+상태 안내음 전용 예약 채널
        self._announce_queue: list = []                  # 순차 재생 대기열 (Sound 객체)
        self._open_status: str | None = None              # 상태어를 아직 안 붙인 진행 중 배치("warn30"/"done")
        self._prev_remaining: dict[int, float] = {}
        self._prev_states: dict[int, BurnerState] = {}
        self._cam_offline_since: dict[int, float] = {}
        self._cam_lost_played: dict[int, bool] = {}
        self._sorted_burners_cache = []

        # 카메라 박스 표시 토글 (V) — 끄면 카메라 없이 화구 카드만
        self._video_hidden: bool = False
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
        self._font_path = _resolve_font_path()
        self._font_cache: dict[int, pygame.font.Font] = {}

        # Pygame 오디오 믹서 초기화 (사운드 에셋 로드)
        try:
            pygame.mixer.init()
            # 채널 0번을 예약 → 번호+상태 안내음 전용. 자동할당(.play())이 가져가지 않도록 보호
            pygame.mixer.set_reserved(1)
            self._announce_channel = pygame.mixer.Channel(0)

            sound_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "sounds")

            def _load(name: str):
                path = os.path.join(sound_dir, name)
                if not os.path.exists(path):
                    path = os.path.join("assets", "sounds", name)
                return pygame.mixer.Sound(path) if os.path.exists(path) else None

            self._sound_warn_30s = _load("30초+전.mp3")
            self._sound_complete = _load("완료.mp3")
            self._sound_cam_lost = _load("cam_lost.wav")
            self._sound_numbers  = {n: s for n in range(1, 31) if (s := _load(f"{n}번.mp3")) is not None}
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
        """저장 정책: 카메라 화면상 배치 순서(좌상단→우하단) + 같은 카메라 안에선 list 위치(=그린 순서) 유지.

        주의: source_id는 카메라가 추가된 순서(내부 식별자)일 뿐 화면에 보이는 위치와 무관함
        (카메라는 배치 모드(L)에서 자유롭게 이동 가능). source_id로 정렬하면 사용자가 화면에서
        "첫 번째 카메라"라고 인식하는 카메라와 실제 ID 1번을 받는 카메라가 어긋나는 버그가
        있었음 — 예: 화면상 1번째 카메라에 1~5번을 그렸는데 2번째 카메라가 source_id가 더
        작다는 이유로 1~5번을 가져가 버림. 화면 배치(layout_pos) 기준으로 정렬해 사용자가
        보는 순서와 부여되는 번호가 항상 일치하도록 한다.

        결과 예시 (화면 좌상단부터):
            화면 1번째 카메라 화구 2개 → ID 1, 2
            화면 2번째 카메라 화구 3개 → ID 3, 4, 5
            화면 3번째 카메라 화구 1개 → ID 6

        선택된 화구 인덱스도 재정렬 후 동일 화구를 가리키도록 갱신.
        """
        selected_obj = (
            self._calib_burners[self._calib_selected_idx]
            if self._calib_selected_idx is not None and self._calib_selected_idx < len(self._calib_burners)
            else None
        )

        # 화면 배치(row, col) 오름차순 정렬 (안정 정렬이라 같은 카메라 안의 그린 순서는 보존됨)
        self._calib_burners.sort(key=lambda b: self._camera_layout_pos(b.get("source_id", 0)))

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

    # ── 통합 캔버스 배치 모드 (카메라 박스 + 화구 카드, 반칸 단위) ──────────
    # 화구 카드 = 2×2 반칸 고정 크기. 카메라 박스 = source별 layout_size (반칸 단위).
    def _default_grid_pos(self, bid: int) -> list[int]:
        """grid_pos 미지정 화구의 기본 위치 — 등장 순서로 2반칸씩 가로로 배치."""
        burners = self.config_data.get("burners", [])
        idx = next((i for i, b in enumerate(burners) if b["id"] == bid), 0)
        cols_per_row = max(1, self._layout_cols // 2)
        row_group = idx // cols_per_row
        idx_in_row = idx % cols_per_row
        return [row_group * 2, idx_in_row * 2]

    def _burner_grid_pos(self, bid: int) -> list[int]:
        cfg = next((b for b in self.config_data.get("burners", []) if b["id"] == bid), None)
        if cfg and "grid_pos" in cfg:
            r, c = cfg["grid_pos"]
            return [int(r), int(c)]
        return self._default_grid_pos(bid)

    def _camera_layout_size(self, src_id: int) -> list[int]:
        sc = next((s for s in self.config_data.get("sources", []) if s["id"] == src_id), None)
        if sc and "layout_size" in sc:
            r, c = sc["layout_size"]
            return [int(r), int(c)]
        return list(_DEFAULT_CAM_SIZE)

    def _default_camera_pos(self, src_id: int) -> list[int]:
        """layout_pos 미지정 카메라 기본 위치 — 화구 영역 아래쪽부터, 그리드 폭(layout_cols)을
        넘기지 않도록 줄바꿈하며 배치."""
        sources = self.config_data.get("sources", [])
        idx = next((i for i, s in enumerate(sources) if s["id"] == src_id), 0)
        cam_h, cam_w = _DEFAULT_CAM_SIZE
        base_row = 8  # 화구 기본 배치 영역(약 4줄) 아래 여유
        cols_per_row = max(1, self._layout_cols // cam_w)
        row_group = idx // cols_per_row
        idx_in_row = idx % cols_per_row
        return [base_row + row_group * cam_h, idx_in_row * cam_w]

    def _camera_layout_pos(self, src_id: int) -> list[int]:
        sc = next((s for s in self.config_data.get("sources", []) if s["id"] == src_id), None)
        if sc and "layout_pos" in sc:
            r, c = sc["layout_pos"]
            return [int(r), int(c)]
        return self._default_camera_pos(src_id)

    def start_layout_mode(self) -> None:
        self.layout_mode = True
        self._layout_positions = {
            bsm.burner_id: self._burner_grid_pos(bsm.burner_id)
            for bsm in self._registry.all()
        }
        self._layout_camera_positions = {
            sc["id"]: self._camera_layout_pos(sc["id"])
            for sc in self.config_data.get("sources", [])
        }
        self._layout_camera_sizes = {
            sc["id"]: self._camera_layout_size(sc["id"])
            for sc in self.config_data.get("sources", [])
        }
        self._layout_drag_kind = None
        self._layout_drag_id = None
        self._selected_id = None
        print("[UI] 배치 모드 시작")

    def cancel_layout_mode(self) -> None:
        self.layout_mode = False
        self._layout_drag_kind = None
        self._layout_drag_id = None

    def save_layout(self) -> None:
        self.layout_mode = False
        self._layout_drag_kind = None
        self._layout_drag_id = None
        for b in self.config_data.get("burners", []):
            pos = self._layout_positions.get(b["id"])
            if pos is not None:
                b["grid_pos"] = [int(pos[0]), int(pos[1])]
        for sc in self.config_data.get("sources", []):
            pos = self._layout_camera_positions.get(sc["id"])
            if pos is not None:
                sc["layout_pos"] = [int(pos[0]), int(pos[1])]
            size = self._layout_camera_sizes.get(sc["id"])
            if size is not None:
                sc["layout_size"] = [int(size[0]), int(size[1])]
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        print("[UI] 배치 저장 완료")
        if getattr(self, "on_config_reloaded", None):
            self.on_config_reloaded(self.config_data)

    def _layout_cell_at(
        self, pos: tuple[int, int], size: tuple[int, int] = (1, 1),
    ) -> tuple[int, int] | None:
        """픽셀 좌표(좌측상단 기준 앵커) → (row, col) 반칸 그리드 좌표. 캔버스 원점/반칸크기 기준.

        size=(rows, cols): 배치할 오브젝트의 반칸 크기. 오브젝트가 그리드 밖으로
        삐져나가 화면 밖에서 찾을 수 없게 되는 일을 막기 위해, 우측/하단 경계 안에
        들어오도록 좌표를 함께 clamp한다.
        """
        if self._half_cell_w <= 0 or self._half_cell_h <= 0:
            return None
        size_rows, size_cols = size
        gx, gy = self._canvas_origin
        rel_x = max(0, pos[0] - gx)
        rel_y = max(0, pos[1] - gy)
        col = int(rel_x // self._half_cell_w)
        row = int(rel_y // self._half_cell_h)
        max_col = max(0, self._layout_cols - size_cols)
        max_row = max(0, self._layout_rows - size_rows)
        col = max(0, min(max_col, col))
        row = max(0, min(max_row, row))
        return (row, col)

    def _layout_drag_anchor(self) -> tuple[int, int]:
        """드래그 중인 오브젝트의 좌측상단 앵커 픽셀 좌표 (그랩 오프셋 보정, 비스냅)."""
        ox, oy = self._layout_drag_offset
        return (self._layout_drag_pos[0] - ox, self._layout_drag_pos[1] - oy)

    def _layout_resize_dims(self, src_id: int, drag_pos: tuple[int, int]) -> list[int]:
        """카메라 박스 좌측상단 기준으로 drag_pos까지의 거리를 반칸 단위 (rows, cols)로 환산. 최소 2x2."""
        if self._half_cell_w <= 0 or self._half_cell_h <= 0:
            return list(self._camera_layout_size(src_id))
        gx, gy = self._canvas_origin
        row, col = self._layout_camera_positions.get(src_id, self._camera_layout_pos(src_id))
        ox = gx + col * self._half_cell_w
        oy = gy + row * self._half_cell_h
        max_cols = max(2, self._layout_cols - col)
        max_rows = max(2, self._layout_rows - row)
        cols = max(2, min(max_cols, round((drag_pos[0] - ox) / self._half_cell_w)))
        rows = max(2, min(max_rows, round((drag_pos[1] - oy) / self._half_cell_h)))
        return [rows, cols]

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

        if self.layout_mode:
            # 배치 모드 전용 키
            if key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
                self.save_layout()
            elif key == pygame.K_ESCAPE:
                self.cancel_layout_mode()
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
        elif key == pygame.K_l:
            self.start_layout_mode()
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
        if self.layout_mode:
            # 리사이즈 핸들 우선 확인 (카메라 박스 모서리)
            for src_id, rect in self._cam_resize_handles.items():
                if rect.collidepoint(pos):
                    self._layout_drag_kind = 'resize'
                    self._layout_drag_id = src_id
                    self._layout_drag_pos = pos
                    self._layout_drag_offset = (0, 0)
                    return
            for bid, rect in self._card_rects.items():
                if rect.collidepoint(pos):
                    self._layout_drag_kind = 'burner'
                    self._layout_drag_id = bid
                    self._layout_drag_pos = pos
                    self._layout_drag_offset = (pos[0] - rect.x, pos[1] - rect.y)
                    return
            for src_id, rect in self._cam_box_rects.items():
                if rect.collidepoint(pos):
                    self._layout_drag_kind = 'camera'
                    self._layout_drag_id = src_id
                    self._layout_drag_pos = pos
                    self._layout_drag_offset = (pos[0] - rect.x, pos[1] - rect.y)
                    return
            return

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

        # 소리 토글 버튼 클릭 (px 영역 정의 필요하나 렌더링 때 저장해 둔 Rect 사용)
        if self._sound_rect and self._sound_rect.collidepoint(pos):
            self.mute = not self.mute
            self._show_toast("음소거 ON" if self.mute else "음소거 OFF")
            return

        # 배치 모드 버튼 클릭
        if getattr(self, "_layout_btn_rect", None) and self._layout_btn_rect.collidepoint(pos):
            self.start_layout_mode()
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
        if self.layout_mode and self._layout_drag_kind is not None:
            self._layout_drag_pos = pos
            return
        if self.calibration_mode and self._calib_dragging:
            v_pos = self._to_video_pos(pos)
            if v_pos and v_pos[0] == self._calib_active_source:
                # 드래그가 다른 카메라 셀로 넘어가는 건 무시 (시작 셀 안에서만)
                self._calib_drag_end = (v_pos[1], v_pos[2])

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self.layout_mode and self._layout_drag_kind is not None:
            if self._layout_drag_kind == 'resize':
                sid = self._layout_drag_id
                self._layout_camera_sizes[sid] = self._layout_resize_dims(sid, pos)
            else:
                ox, oy = self._layout_drag_offset
                anchor = (pos[0] - ox, pos[1] - oy)
                if self._layout_drag_kind == 'burner':
                    bid = self._layout_drag_id
                    target = self._layout_cell_at(anchor, size=(2, 2))
                    if target is not None:
                        occupant = next(
                            (b for b, p in self._layout_positions.items()
                             if b != bid and tuple(p) == target),
                            None,
                        )
                        if occupant is not None:
                            self._layout_positions[occupant] = list(self._layout_positions[bid])
                        self._layout_positions[bid] = [target[0], target[1]]
                elif self._layout_drag_kind == 'camera':
                    sid = self._layout_drag_id
                    cam_size = self._layout_camera_sizes.get(sid, self._camera_layout_size(sid))
                    target = self._layout_cell_at(anchor, size=tuple(cam_size))
                    if target is not None:
                        self._layout_camera_positions[sid] = [target[0], target[1]]
            self._layout_drag_kind = None
            self._layout_drag_id = None
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
                        "countdown_second": 270,
                        "done_first_timeout": 120,
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

        self._flush_announcements()

        # 길게 누르기 버튼 처리
        now = time.monotonic()
        for bid, start_t in list(self._reset_hold.items()):
            if now - start_t >= _RESET_HOLD_S:
                self._registry.get(bid).manual_reset()
                del self._reset_hold[bid]

        sw, sh = self._screen.get_size()
        self._screen.fill(_C_BG)

        canvas_y0 = self._draw_toolbar(sw)
        canvas_h = max(0, sh - canvas_y0)

        # 통합 캔버스 좌표계 (반칸 그리드) 갱신 — 매 프레임 창 크기에 맞춰 재계산
        self._canvas_origin = (0, canvas_y0)
        self._half_cell_w = max(1, sw // max(1, self._layout_cols))
        self._half_cell_h = max(1, int(self._half_cell_w * _HALF_CELL_ASPECT))

        if self.calibration_mode:
            self._draw_canvas_cameras(frames or {}, processor, sw, canvas_y0, canvas_h, force_all=True)
            self._draw_calibration_overlay(sw, canvas_h, canvas_y0)
        elif self.layout_mode:
            self._draw_layout_editor(frames or {}, processor, sw, canvas_y0, canvas_h)
        else:
            self._draw_canvas_cameras(frames or {}, processor, sw, canvas_y0, canvas_h, force_all=False)
            self._draw_canvas_burners()

        # Toast (있으면)
        if self._toast_msg and time.monotonic() < self._toast_until:
            toast = self._fonts["label"].render(self._toast_msg, True, _C_TEXT_DARK)
            pad = 14
            box = pygame.Rect(0, 0, toast.get_width() + pad * 2, toast.get_height() + 12)
            box.midbottom = (sw // 2, sh - 24)
            pygame.draw.rect(self._screen, _C_BRAND, box, border_radius=6)
            self._screen.blit(toast, toast.get_rect(center=box.center))

        pygame.display.flip()
        if self._clock:
            self._clock.tick(30) # 30fps UI Redraw

    def _draw_canvas_cameras(
        self, frames: dict, processor, sw: int, canvas_y0: int, canvas_h: int, *, force_all: bool,
    ) -> None:
        """카메라 박스들을 반칸 그리드의 layout_pos/layout_size 위치에 그린다.

        force_all=True (캘리브레이션 모드): 풀뷰/영상숨김 무시하고 전체 카메라 표시.
        풀뷰(_focus_camera_id) 지정 시 해당 카메라만 캔버스 전체로 크게 표시.
        오프라인 추적은 항상 전체 카메라 대상으로 수행.
        """
        all_sources = self.config_data.get("sources", [])
        if not all_sources:
            self._cam_cells.clear()
            return

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

        if self._video_hidden and not force_all:
            self._cam_cells.clear()
            return

        self._cam_cells.clear()
        self._cam_box_rects.clear()

        if not force_all and self._focus_camera_id is not None:
            focused = [s for s in all_sources if s["id"] == self._focus_camera_id]
            if not focused:
                self._focus_camera_id = None
            else:
                sc = focused[0]
                margin = 20
                ox, oy = margin, canvas_y0 + margin
                cw, ch = max(1, sw - margin * 2), max(1, canvas_h - margin * 2)
                cell_bg = pygame.Rect(ox, oy, cw, ch)
                pygame.draw.rect(self._screen, _C_PANEL, cell_bg, border_radius=4)
                self._draw_camera_cell(sc, frames.get(sc["id"]), processor, ox, oy, cw, ch)
                hint = self._fonts["small"].render("더블클릭으로 그리드 복귀", True, _C_BRAND)
                self._screen.blit(hint, (ox + 8, oy + ch - 22))
                return

        gx, gy = self._canvas_origin
        for sc in all_sources:
            src_id = sc["id"]
            row, col = self._camera_layout_pos(src_id)
            rows, cols = self._camera_layout_size(src_id)
            ox = gx + col * self._half_cell_w
            oy = gy + row * self._half_cell_h
            cw = max(1, cols * self._half_cell_w)
            ch = max(1, rows * self._half_cell_h)

            cell_bg = pygame.Rect(ox, oy, cw, ch)
            self._cam_box_rects[src_id] = cell_bg
            pygame.draw.rect(self._screen, _C_PANEL, cell_bg, border_radius=4)

            frame = frames.get(src_id)
            self._draw_camera_cell(sc, frame, processor, ox, oy, cw, ch)

            is_lost = False
            if src_id in self._cam_offline_since:
                if now - self._cam_offline_since[src_id] >= 3.0:
                    is_lost = True
                    if not self.mute and self._sound_cam_lost:
                        if not self._cam_lost_played.get(src_id, False):
                            self._sound_cam_lost.play()
                            self._cam_lost_played[src_id] = True

            if is_lost:
                if int(time.time() * 3) % 2 == 0:
                    pygame.draw.rect(self._screen, _C_FAILED, cell_bg, width=4, border_radius=4)
            elif self._selected_camera_id == src_id:
                pygame.draw.rect(self._screen, _C_BRAND, cell_bg, width=4, border_radius=4)

    def _draw_canvas_burners(self) -> None:
        """화구 카드들을 반칸 그리드의 grid_pos 위치(2×2 반칸 고정 크기)에 그린다."""
        self._card_rects.clear()
        self._start_rects.clear()
        self._reset_rects.clear()

        gx, gy = self._canvas_origin
        w = 2 * self._half_cell_w
        h = 2 * self._half_cell_h
        for bsm in self._registry.all():
            row, col = self._burner_grid_pos(bsm.burner_id)
            x = gx + col * self._half_cell_w
            y = gy + row * self._half_cell_h
            self._draw_burner_card(bsm, x, y, w, h)

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

    def _draw_calibration_overlay(self, box_w, box_h, canvas_y0: int = 0):
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

        # 3. 상단 안내 바 (캔버스 영역 상단에 오버레이)
        banner = pygame.Surface((box_w, 60), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 180))
        self._screen.blit(banner, (0, canvas_y0))
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
        self._screen.blit(title, (20, canvas_y0 + 8))
        self._screen.blit(desc, (20, canvas_y0 + 36))

    def _draw_toolbar(self, sw: int) -> int:
        """상단 통합 툴바 (헤더 + 버튼 + 카메라 관리). 반환값 = 캔버스 시작 y."""
        header_h = 50
        pygame.draw.rect(self._screen, _C_BRAND, (0, 0, sw, header_h))
        title_surf = self._fonts["title"].render("길가옆에 압력밥솥 타이머", True, _C_TEXT_DARK)
        self._screen.blit(title_surf, (16, 14))

        oy = header_h + 6
        if self._model_missing:
            wrng = pygame.Rect(10, oy, sw - 20, 30)
            pygame.draw.rect(self._screen, _C_WARNING_BG, wrng, border_radius=6)
            msg = self._fonts["small"].render("⚠ AI 모델 없음 (수동 구동만 가능)", True, _C_TEXT_LIGHT)
            self._screen.blit(msg, (20, oy + 7))
            oy += 38

        toolbar_h = 40
        toolbar_rect = pygame.Rect(10, oy, sw - 20, toolbar_h)
        pygame.draw.rect(self._screen, _C_CARD_BG, toolbar_rect, border_radius=6)
        by = oy + 12

        conf_btn   = self._fonts["small"].render("⚙ 설정(F)", True, _C_BRAND if not self.calibration_mode else _C_TEXT_LIGHT)
        layout_btn = self._fonts["small"].render("🪄 배치(L)", True, _C_BRAND if not self.layout_mode else _C_TEXT_LIGHT)
        mask_btn   = self._fonts["small"].render(f"Mask(M): {'ON' if self.show_mask else 'OFF'}", True, _C_BRAND if self.show_mask else _C_TEXT_LIGHT)
        cam_btn    = self._fonts["small"].render("카메라 전환(C)", True, _C_TEXT_LIGHT)
        dev_btn    = self._fonts["small"].render(f"Dev(D): {'ON' if self.dev_mode else 'OFF'}", True, _C_BRAND if self.dev_mode else _C_TEXT_LIGHT)
        play_btn   = self._fonts["small"].render("일시정지(Space)" if not self.video_paused else "재생(Space)", True, _C_BRAND if self.video_paused else _C_TEXT_LIGHT)
        hide_btn   = self._fonts["small"].render(f"영상(V): {'숨김' if self._video_hidden else '표시'}", True, _C_BRAND if self._video_hidden else _C_TEXT_LIGHT)
        sound_btn  = self._fonts["small"].render(f"소리(N): {'OFF' if self.mute else 'ON'}", True, _C_BRAND if not self.mute else _C_FAILED)

        xs = [20, 110, 200, 300, 420, 500, 620, 710]
        items = [conf_btn, layout_btn, mask_btn, cam_btn, dev_btn, play_btn, hide_btn, sound_btn]
        for surf, x in zip(items, xs):
            self._screen.blit(surf, (x, by))

        self._layout_btn_rect = pygame.Rect(xs[1], oy + 6, 80, 28)
        self._sound_rect      = pygame.Rect(xs[7], oy + 6, 90, 28)

        # 카메라 대수 + +/- 버튼 (우측 정렬)
        n_cams = len(self.config_data.get("sources", []))
        sel_hint = f" (선택:Cam-{self._selected_camera_id})" if self._selected_camera_id is not None else ""
        cam_label = self._fonts["small"].render(f"카메라 {n_cams}대{sel_hint}", True, _C_TEXT_LIGHT)

        btn_w, btn_h = 64, 26
        add_x = sw - 16 - btn_w
        rem_x = add_x - btn_w - 6
        label_x = rem_x - cam_label.get_width() - 16
        self._screen.blit(cam_label, (label_x, by))

        self._cam_remove_rect = pygame.Rect(rem_x, oy + 7, btn_w, btn_h)
        self._cam_add_rect    = pygame.Rect(add_x, oy + 7, btn_w, btn_h)
        rem_active = self._selected_camera_id is not None
        rem_color = (180, 60, 60) if rem_active else (60, 60, 70)
        rem_text = f"−Cam{self._selected_camera_id}" if rem_active else "− 제거"
        pygame.draw.rect(self._screen, rem_color, self._cam_remove_rect, border_radius=4)
        pygame.draw.rect(self._screen, _C_SUCCESS, self._cam_add_rect, border_radius=4)
        rem_surf = self._fonts["btn"].render(rem_text, True, _C_TEXT_LIGHT)
        add_surf = self._fonts["btn"].render("+추가", True, _C_TEXT_LIGHT)
        self._screen.blit(rem_surf, rem_surf.get_rect(center=self._cam_remove_rect.center))
        self._screen.blit(add_surf, add_surf.get_rect(center=self._cam_add_rect.center))

        oy += toolbar_h + 8
        return oy

    def _draw_layout_editor(
        self, frames: dict, processor, sw: int, canvas_y0: int, canvas_h: int,
    ) -> None:
        """배치 모드 — 카메라 박스 + 화구 카드를 반칸 그리드 위에서 드래그로 재배치."""
        gx, gy = self._canvas_origin
        cols = max(1, self._layout_cols)
        rows = max(1, canvas_h // self._half_cell_h) if self._half_cell_h else 1
        self._layout_rows = rows

        # 그리드 라인
        for c in range(cols + 1):
            x = gx + c * self._half_cell_w
            pygame.draw.line(self._screen, (60, 60, 70), (x, gy), (x, gy + rows * self._half_cell_h), 1)
        for r in range(rows + 1):
            y = gy + r * self._half_cell_h
            pygame.draw.line(self._screen, (60, 60, 70), (gx, y), (gx + cols * self._half_cell_w, y), 1)

        # 카메라 박스 (드래그/리사이즈 가능) — 항상 좌측상단 기준으로 스냅된 실제 위치/크기를 그린다
        self._cam_cells.clear()
        self._cam_box_rects.clear()
        self._cam_resize_handles.clear()
        for sc in self.config_data.get("sources", []):
            src_id = sc["id"]
            row, col = self._layout_camera_positions.get(src_id, self._camera_layout_pos(src_id))
            rsize, csize = self._layout_camera_sizes.get(src_id, self._camera_layout_size(src_id))
            ox = gx + col * self._half_cell_w
            oy_box = gy + row * self._half_cell_h
            cw = max(1, csize * self._half_cell_w)
            ch = max(1, rsize * self._half_cell_h)

            is_active = self._layout_drag_id == src_id
            if is_active and self._layout_drag_kind == 'camera':
                anchor = self._layout_drag_anchor()
                target = self._layout_cell_at(anchor, size=(rsize, csize))
                if target is not None:
                    ox = gx + target[1] * self._half_cell_w
                    oy_box = gy + target[0] * self._half_cell_h
            elif is_active and self._layout_drag_kind == 'resize':
                live_rows, live_cols = self._layout_resize_dims(src_id, self._layout_drag_pos)
                cw = max(1, live_cols * self._half_cell_w)
                ch = max(1, live_rows * self._half_cell_h)

            cell_bg = pygame.Rect(ox, oy_box, cw, ch)
            self._cam_box_rects[src_id] = cell_bg
            pygame.draw.rect(self._screen, _C_PANEL, cell_bg, border_radius=4)
            frame = (frames or {}).get(src_id)
            self._draw_camera_cell(sc, frame, processor, ox, oy_box, cw, ch)
            # 오프라인이라 _cam_cells에 못 등록됐을 경우를 위한 드래그용 폴백 rect
            self._cam_cells.setdefault(src_id, {"rect": cell_bg, "scale": 1.0, "frame_size": (cw, ch)})
            border_color = _C_BRAND if is_active and self._layout_drag_kind in ('camera', 'resize') else (120, 200, 255)
            border_w = 3 if is_active else 2
            pygame.draw.rect(self._screen, border_color, cell_bg, width=border_w, border_radius=4)

            # 리사이즈 핸들 (우하단 모서리, ↘ 모양)
            hs = 14
            handle_rect = pygame.Rect(cell_bg.right - hs, cell_bg.bottom - hs, hs, hs)
            self._cam_resize_handles[src_id] = handle_rect
            pygame.draw.rect(self._screen, _C_BRAND, handle_rect, border_radius=3)
            pygame.draw.line(
                self._screen, _C_TEXT_DARK,
                (handle_rect.x + 3, handle_rect.bottom - 3), (handle_rect.right - 3, handle_rect.y + 3), 2,
            )

        # 화구 카드 (드래그 가능) — 좌측상단 기준으로 스냅된 실제 위치를 그린다
        self._card_rects.clear()
        self._start_rects.clear()
        self._reset_rects.clear()
        w = 2 * self._half_cell_w
        h = 2 * self._half_cell_h
        for bsm in self._registry.all():
            bid = bsm.burner_id
            row, col = self._layout_positions.get(bid, [0, 0])
            x = gx + col * self._half_cell_w
            y = gy + row * self._half_cell_h
            is_dragging = self._layout_drag_kind == 'burner' and self._layout_drag_id == bid
            if is_dragging:
                anchor = self._layout_drag_anchor()
                target = self._layout_cell_at(anchor, size=(2, 2))
                if target is not None:
                    x = gx + target[1] * self._half_cell_w
                    y = gy + target[0] * self._half_cell_h
            self._draw_burner_card(bsm, x, y, w, h)
            if is_dragging:
                pygame.draw.rect(self._screen, _C_BRAND, (x, y, w, h), width=3, border_radius=8)

        # 안내 배너
        banner = pygame.Surface((sw, 36), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 180))
        self._screen.blit(banner, (0, canvas_y0))
        title = self._fonts["small"].render(
            "🪄 배치 모드 — 드래그=이동, 카메라 모서리 핸들=크기조절   ENTER=저장  ESC=취소",
            True, _C_BRAND,
        )
        self._screen.blit(title, (10, canvas_y0 + 9))

    @staticmethod
    def _card_text_color(bg_color: tuple) -> tuple:
        r, g, b = bg_color
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return _C_TEXT_DARK if lum > 0.4 else _C_TEXT_LIGHT

    def _dynamic_font(self, size: int) -> pygame.font.Font:
        """카드 크기에 비례하는 화구 번호 폰트 — 크기별 캐시."""
        size = max(8, int(size))
        f = self._font_cache.get(size)
        if f is None:
            f = pygame.font.Font(self._font_path, size) if self._font_path else pygame.font.Font(None, size + 4)
            f.set_bold(True)
            self._font_cache[size] = f
        return f

    def _queue_announcement(self, status: str, bid: int) -> None:
        """화구 안내 이벤트를 지연 없이 즉시 재생열에 적재.

        진행 중인 배치(아직 상태어를 안 붙인 번호들)가 같은 상태(warn30/done)면
        번호만 이어붙인다 — "1번 2번 3번 30초전"처럼 합쳐짐.
        다른 상태가 끼어들면 진행 중인 배치를 즉시 상태어로 마무리하고 새 배치를 연다.
        """
        if self._open_status is not None and self._open_status != status:
            self._seal_open_batch()
        self._open_status = status
        snd = self._sound_numbers.get(bid)
        if snd:
            self._announce_queue.append(snd)

    def _seal_open_batch(self) -> None:
        """열려 있는 배치에 상태어를 붙여 마무리."""
        status = self._open_status
        self._open_status = None
        if status is None:
            return
        status_snd = self._sound_warn_30s if status == "warn30" else self._sound_complete
        if status_snd:
            self._announce_queue.append(status_snd)

    def _flush_announcements(self) -> None:
        """예약 채널이 비면 다음 항목 재생. 더 재생할 게 없는데 배치가 열려 있으면
        (재생 시간 동안 같은 상태의 다른 화구가 끼어들지 않았다는 뜻) 그 자리에서 마무리."""
        if self.mute:
            self._announce_queue.clear()
            self._open_status = None
            return
        if self._announce_channel is None:
            return
        if not self._announce_channel.get_busy():
            if not self._announce_queue and self._open_status is not None:
                self._seal_open_batch()
            if self._announce_queue:
                self._announce_channel.play(self._announce_queue.pop(0))

    def _draw_burner_card(self, bsm, x, y, w, h):
        bid = bsm.burner_id
        selected = (bid == self._selected_id)

        # 알림음 재생 제어 (작업 E)
        cur_rem = bsm.remaining_seconds
        prev_rem = self._prev_remaining.get(bid, cur_rem)

        # 1) 완료 30초 전 예고 (남은 시간이 30초 선을 막 통과했을 때)
        from core.state_machine import BurnerState as BS
        if bsm.is_counting and prev_rem > 30 >= cur_rem > 0:
            self._queue_announcement("warn30", bid)

        # 2) 완료 순간 전이 (STEAMING -> DONE_FIRST 또는 DONE_SECOND)
        prev_st = self._prev_states.get(bid, bsm.state)
        if prev_st in (BS.POT_STEAMING_FIRST, BS.POT_STEAMING_SECOND) and bsm.state in (BS.DONE_FIRST, BS.DONE_SECOND):
            self._queue_announcement("done", bid)

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

        # --- 1. 화구 번호 — 카드에서 가장 크고 눈에 잘 보이는 핵심 요소.
        # 상태/타이머/게이지/버튼이 들어갈 하단 footer를 제외한 나머지 영역을
        # 전부 채우는 크기로 렌더링하고, 가로 중앙에 배치한다.
        footer_h = 54  # 상태줄 + 게이지 + 버튼 영역에 필요한 고정 높이
        if self.dev_mode:
            # 개발 모드: 정보가 많아 번호는 콤팩트한 배지로 축소
            rad = 8
            cx, cy = x + 10, y + 11
            pygame.draw.circle(self._screen, _C_BRAND, (cx, cy), rad)
            id_surf = self._fonts["small"].render(str(bid), True, _C_TEXT_DARK)
            self._screen.blit(id_surf, id_surf.get_rect(center=(cx, cy)))
            num_zone_h = 20
        else:
            num_str = f"{bid:02d}" if bid >= 10 else str(bid)
            num_zone_h = max(20, h - footer_h)
            num_size = max(22, int(num_zone_h * 0.85))
            num_font = self._dynamic_font(num_size)
            id_surf = num_font.render(num_str, True, tc)
            id_x = x + (w - id_surf.get_width()) // 2
            id_y = y + 2 + max(0, (num_zone_h - id_surf.get_height()) // 2)
            self._screen.blit(id_surf, (id_x, id_y))
        body_y = y + num_zone_h + 2

        # --- 3. 상태 메시지 및 타이머 — 번호 아래 보조 정보 한 줄 ---
        if bsm.state in (BS.EMPTY, BS.POT_IDLE, BS.WAIT_SECOND, BS.DONE_SECOND):
            status_text = {
                BS.EMPTY: "대기",
                BS.POT_IDLE: "준비",
                BS.WAIT_SECOND: "재벌대기",
                BS.DONE_SECOND: "완료!",
            }.get(bsm.state, "대기")
            st_surf = self._fonts["card_status"].render(status_text, True, tc)
            self._screen.blit(st_surf, (x + 8, body_y))
        else:
            # 타이머 작동 중인 상태 (STEAMING_FIRST, STEAMING_SECOND, DONE_FIRST) — 단계+시간 한 줄로 결합
            ph_label = "냉각중" if bsm.state == BS.DONE_FIRST else bsm.phase_label
            t_color = _C_BRAND if bsm.state == BS.DONE_FIRST else tc
            combined = f"{ph_label} {bsm.status_label}"
            f_timer = self._fonts["label"] if self.dev_mode else self._fonts["card_status"]
            time_surf = f_timer.render(combined, True, t_color)
            self._screen.blit(time_surf, (x + 8, body_y))

        # --- 4. 개발자 모드 전용 RMS 수치 오버레이 (일반 모드는 완전 격리) ---
        if self.dev_mode:
            rms_val = bsm.current_angle
            rms_text = f"RMS: {rms_val:.3f}" if rms_val is not None else "RMS: -"
            rms_surf = self._fonts["small"].render(rms_text, True, tc)
            self._screen.blit(rms_surf, (x + 8, body_y + 16))

        # --- 5. 진동 게이지 (슬림하고 예쁜 라운드 게이지바) ---
        gauge_x, gauge_y = x + 8, y + h - 34
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

def _resolve_font_path() -> str | None:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/NanumGothic.ttf"
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def _load_fonts() -> dict:
    font_path = _resolve_font_path()
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
