"""
pygame 타이머 UI — Phase 1

화구 카드 그리드 렌더링 + 수동 조작.

화구 카드 레이아웃:
  ┌──────────┐
  │   1번    │  ← 화구 번호
  │  3:42    │  ← 타이머 / 상태
  │ [⟳] [▶] │  ← 초기화 / 수동시작 버튼
  └──────────┘

수동 조작:
  - 카드 클릭        → 화구 선택 (테두리 강조)
  - ⟳ 버튼 길게(1s) → 초기화 (DONE 상태는 즉시)
  - ▶ 버튼 클릭     → 수동 타이머 시작
  - 키 R             → 선택 화구 초기화
  - 키 S             → 선택 화구 수동 시작
  - 키 1~9, 0        → 1~10번 화구 선택
  - ESC              → 선택 해제
"""

from __future__ import annotations

import time

import pygame

from core.state_machine import BurnerRegistry, BurnerState, _STEAMING

# ── 레이아웃 상수 ────────────────────────────────────────────────────────────
_PAD          = 6
_BTN_H        = 28
_TITLE_H      = 36
_WARNING_H    = 28
_STATUS_BAR_H = 44   # 상태바 높이 (2줄: 단축키 안내 + 상태 선례)
_RESET_HOLD_S = 1.0   # 초기화 버튼 길게 누르기 필요 시간 (초)

# 색상
_C_BG         = (18,  18,  18)
_C_TITLE      = (220, 220, 220)
_C_WARNING_BG = (180,  60,  10)
_C_WARNING_FG = (255, 255, 255)
_C_SELECTED   = (255, 220,  30)
_C_BTN_RESET  = (60,  60,  140)
_C_BTN_START  = (40, 140,  60)
_C_BTN_HOLD   = (200, 100,  20)
_C_BTN_TEXT   = (255, 255, 255)
_C_CARD_BORDER = (50,  50,  50)


class UIDisplay:
    """pygame 기반 화구 그리드 UI"""

    def __init__(
        self,
        ui_cfg:      dict,
        registry:    BurnerRegistry,
        burner_meta: dict,          # {burner_id: {grid_pos: [row, col]}}
        model_missing: bool = True,
    ):
        self._cfg          = ui_cfg
        self._registry     = registry
        self._meta         = burner_meta
        self._model_missing = model_missing

        self._screen: pygame.Surface | None = None
        self._clock:  pygame.time.Clock | None = None
        self._fonts:  dict | None = None

        # 선택 상태
        self._selected_id: int | None = None

        # 초기화 버튼 길게 누르기
        self._reset_hold:  dict[int, float] = {}  # {burner_id: press_start}

        # 버튼 히트박스 (render 중에 계산 후 저장)
        self._card_rects:  dict[int, pygame.Rect] = {}
        self._reset_rects: dict[int, pygame.Rect] = {}
        self._start_rects: dict[int, pygame.Rect] = {}

    # ── 초기화 / 종료 ──────────────────────────────────────────────────

    def init(self) -> None:
        pygame.init()
        w, h  = self._cfg.get("window_size", [1280, 720])
        title = self._cfg.get("window_title", "압력밥솔 타이머")
        # RESIZABLE 플래그 추가 → 마우스로 창 크기 조절 가능
        self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        pygame.display.set_caption(title)
        self._clock  = pygame.time.Clock()
        self._fonts  = _load_fonts()

    def quit(self) -> None:
        pygame.quit()

    # ── 이벤트 처리 ────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> bool:
        """True 반환 → 종료 요청"""
        if event.type == pygame.QUIT:
            return True

        # 상태 포개증 - 유동 스크린 업데이트
        if event.type == pygame.VIDEORESIZE:
            w, h = event.w, event.h
            self._screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            return False

        if event.type == pygame.KEYDOWN:
            return self._on_keydown(event.key)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_mouse_down(event.pos)

        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._on_mouse_up(event.pos)

        return False

    def _on_keydown(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            if self._selected_id is not None:
                self._selected_id = None
                return False
            return True  # 선택 없으면 종료

        # 숫자 키로 화구 선택
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
            return False

        if self._selected_id is None:
            return False

        bsm = self._registry.get(self._selected_id)
        if key == pygame.K_r:
            self._do_reset(self._selected_id, bsm)
        elif key == pygame.K_s:
            bsm.manual_start()

        return False

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        # 초기화 버튼
        for bid, rect in self._reset_rects.items():
            if rect.collidepoint(pos):
                bsm = self._registry.get(bid)
                if bsm.state == BurnerState.DONE_SECOND:
                    bsm.manual_reset()
                else:
                    self._reset_hold[bid] = time.monotonic()
                return

        # 시작 버튼
        for bid, rect in self._start_rects.items():
            if rect.collidepoint(pos):
                self._registry.get(bid).manual_start()
                return

        # 카드 클릭 → 선택
        for bid, rect in self._card_rects.items():
            if rect.collidepoint(pos):
                self._selected_id = bid if self._selected_id != bid else None
                return

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        self._reset_hold.clear()

    def _do_reset(self, bid: int, bsm) -> None:
        bsm.manual_reset()
        self._reset_hold.pop(bid, None)

    # ── 렌더링 ─────────────────────────────────────────────────────────

    def render(self) -> None:
        if self._screen is None:
            return

        self._process_hold()

        self._screen.fill(_C_BG)
        offset_y = 0
        offset_y += self._draw_title()
        if self._model_missing:
            offset_y += self._draw_warning()
        self._draw_grid(offset_y)
        self._draw_status_bar()

        pygame.display.flip()
        if self._clock:
            self._clock.tick(30)

    def _process_hold(self) -> None:
        """길게 누르기 완료 감지"""
        now = time.monotonic()
        for bid, start in list(self._reset_hold.items()):
            if now - start >= _RESET_HOLD_S:
                self._registry.get(bid).manual_reset()
                del self._reset_hold[bid]

    def _draw_title(self) -> int:
        title = self._cfg.get("window_title", "압력밥솥 타이머 모니터")
        surf  = self._fonts["title"].render(title, True, _C_TITLE)
        self._screen.blit(surf, (10, 8))
        return _TITLE_H

    def _draw_warning(self) -> int:
        sw = self._screen.get_width()
        bar = pygame.Rect(0, _TITLE_H, sw, _WARNING_H)
        pygame.draw.rect(self._screen, _C_WARNING_BG, bar)
        msg  = "⚠  모델 없음 — 테스트 모드  (자동 감지 비활성화. 수동 ▶ 버튼으로 타이머 시작)"
        surf = self._fonts["small"].render(msg, True, _C_WARNING_FG)
        self._screen.blit(surf, (10, _TITLE_H + (_WARNING_H - surf.get_height()) // 2))
        return _WARNING_H

    def _draw_grid(self, offset_y: int) -> None:
        burners = sorted(self._registry.all(), key=lambda b: b.burner_id)
        if not burners:
            return

        cols = self._cfg.get("grid_cols", 6)
        sw, sh = self._screen.get_size()

        # grid_pos 에서 최대 행 계산
        max_row = max(
            self._meta.get(b.burner_id, {}).get("grid_pos", [0, 0])[0]
            for b in burners
        )
        rows   = max_row + 1
        cell_w = sw // cols
        # 상태바 높이를 빼서 카드가 버튼을 덮지 않도록 함
        cell_h = max(90, (sh - offset_y - _STATUS_BAR_H) // rows)

        self._card_rects.clear()
        self._reset_rects.clear()
        self._start_rects.clear()

        for bsm in burners:
            gp  = self._meta.get(bsm.burner_id, {}).get("grid_pos", [0, bsm.burner_id - 1])
            row, col = gp[0], gp[1]
            x = col * cell_w
            y = offset_y + row * cell_h
            self._draw_card(bsm, x, y, cell_w, cell_h)

    def _draw_card(self, bsm, x: int, y: int, w: int, h: int) -> None:
        bid   = bsm.burner_id
        color = bsm.color
        selected = (bid == self._selected_id)

        # 카드 배경
        card_rect = pygame.Rect(x + 2, y + 2, w - 4, h - 4)
        self._card_rects[bid] = card_rect
        pygame.draw.rect(self._screen, color, card_rect, border_radius=8)

        # 테두리 (선택 강조)
        border_color = _C_SELECTED if selected else _C_CARD_BORDER
        border_w     = 3 if selected else 1
        pygame.draw.rect(self._screen, border_color, card_rect, border_w, border_radius=8)

        cx = x + w // 2

        # 화구 번호
        id_surf = self._fonts["id"].render(str(bid), True, (255, 255, 255))
        self._screen.blit(id_surf, id_surf.get_rect(centerx=cx, top=y + _PAD + 2))

        # 초벌/재벌 단계 레이블 (타이머 위 소형 텍스트)
        phase = bsm.phase_label
        if phase:
            ph_surf = self._fonts["small"].render(phase, True, (220, 220, 180))
            self._screen.blit(ph_surf, ph_surf.get_rect(centerx=cx, top=y + _PAD + 32))

        # 타이머 / 상태 레이블
        label = bsm.status_label
        font  = self._fonts["timer"] if bsm.state in _STEAMING else self._fonts["label"]
        lbl_surf = font.render(label, True, (255, 255, 255))
        mid_y    = y + h // 2 - _BTN_H // 2 - _PAD
        self._screen.blit(lbl_surf, lbl_surf.get_rect(centerx=cx, centery=mid_y))

        # 딸랑이 감지 인디케이터 (타이머 아래)
        # POT_IDLE / DONE_FIRST 상태에서만 표시 (진동 감지 준비 중일 때)
        if bsm.state in (BurnerState.POT_IDLE, BurnerState.DONE_FIRST):
            bar_w  = w - _PAD * 4
            bar_h  = 6
            bar_x  = x + _PAD * 2
            bar_y  = mid_y + lbl_surf.get_height() // 2 + _PAD + 14
            # 딸랑이 감지 바 레이블
            lbl = self._fonts["small"].render("딸랑이 감지", True, (160, 160, 160))
            self._screen.blit(lbl, (bar_x, bar_y - 14))
            # 배경
            pygame.draw.rect(self._screen, (60, 60, 60),
                             pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=3)
            # 진행도
            fill_w = int(bar_w * min(1.0, bsm.vibration_score))
            if fill_w > 0:
                bar_color = (255, 220, 60) if bsm.vibration_score < 1.0 else (80, 255, 80)
                pygame.draw.rect(self._screen, bar_color,
                                 pygame.Rect(bar_x, bar_y, fill_w, bar_h), border_radius=3)
            # 딸랑이 감지 점 (오른쪽 위)
            dot_color = (255, 220, 60) if bsm.weight_detected else (80, 80, 80)
            pygame.draw.circle(self._screen, dot_color, (x + w - _PAD * 2, y + _PAD + 10), 5)

            # 진동 판정 디버그 표시 (ncc: 정규화 교차상관, 낮을수록 패턴 변화)
            ncc_val = getattr(bsm, "current_angle", None)
            if ncc_val is not None:
                angle_str   = f"ncc {ncc_val:.3f}"
                angle_color = (80, 255, 80) if ncc_val < 0.85 else (255, 220, 60)
            else:
                angle_str   = "ncc --"
                angle_color = (100, 100, 100)
            asurf = self._fonts["small"].render(angle_str, True, angle_color)
            self._screen.blit(asurf, (bar_x, bar_y + bar_h + 4))

        # 하단 버튼 행 — 상태에 따라 레이블 변경
        from core.state_machine import BurnerState as BS
        if bsm.state in (BS.EMPTY, BS.POT_IDLE):
            start_label = "▶ 단추"
        elif bsm.state == BS.POT_STEAMING_FIRST:
            start_label = "⏩ 완료"
        elif bsm.state in (BS.DONE_FIRST,):
            start_label = "⏭ 재벌"
        elif bsm.state == BS.POT_STEAMING_SECOND:
            start_label = "⏩ 완료"
        else:
            start_label = "시작"

        btn_y    = y + h - _BTN_H - _PAD
        btn_w    = (w - _PAD * 3) // 2

        reset_rect = pygame.Rect(x + _PAD,           btn_y, btn_w, _BTN_H)
        start_rect = pygame.Rect(x + _PAD + btn_w + _PAD, btn_y, btn_w, _BTN_H)
        self._reset_rects[bid] = reset_rect
        self._start_rects[bid] = start_rect

        # 초기화 버튼 — 길게 누르는 중이면 진행 표시
        hold_prog = 0.0
        if bid in self._reset_hold:
            elapsed   = time.monotonic() - self._reset_hold[bid]
            hold_prog = min(1.0, elapsed / _RESET_HOLD_S)
        self._draw_button(reset_rect, "초기화", _C_BTN_RESET, hold_prog)
        self._draw_button(start_rect, start_label, _C_BTN_START)

    def _draw_button(
        self,
        rect: pygame.Rect,
        text: str,
        color: tuple,
        hold_progress: float = 0.0,
    ) -> None:
        pygame.draw.rect(self._screen, color, rect, border_radius=5)

        if hold_progress > 0:
            fill = pygame.Rect(rect.x, rect.y, int(rect.w * hold_progress), rect.h)
            pygame.draw.rect(self._screen, _C_BTN_HOLD, fill, border_radius=5)

        surf = self._fonts["btn"].render(text, True, _C_BTN_TEXT)
        self._screen.blit(surf, surf.get_rect(center=rect.center))

    def _draw_status_bar(self) -> None:
        """2줄 상태바: 위쥐 = 단축키/조작 안내, 아래줄 = 상태 선례"""
        sw, sh = self._screen.get_size()
        bar    = pygame.Rect(0, sh - _STATUS_BAR_H, sw, _STATUS_BAR_H)
        pygame.draw.rect(self._screen, (30, 30, 30), bar)
        # 구분선
        pygame.draw.line(self._screen, (60, 60, 60),
                         (0, sh - _STATUS_BAR_H + 22),
                         (sw, sh - _STATUS_BAR_H + 22))

        # 위쥐: 선택 / 조작 안내
        if self._selected_id is not None:
            try:
                self._registry.get(self._selected_id)
            except KeyError:
                self._selected_id = None
        if self._selected_id is not None:
            row1 = f"[{self._selected_id}번 화구]  R: 초기화   S: 수동시작/다음단계   ESC: 선택해제"
        else:
            row1 = "카드 클릭 또는 숫자키(1~0)로 화구 선택  |  ESC: 종료"

        # 아래줄: 상태 선례
        row2 = "■파랑=준비  ■연두=초벌진행  ■노랑=초벌완료/재벌대기  ■진녹=재벌진행  ■빨강=완료 │ 버튼: [초기화] 길게누름(잠금) / [▶단추] 다음단계"

        surf1 = self._fonts["small"].render(row1, True, (200, 200, 200))
        surf2 = self._fonts["small"].render(row2, True, (150, 150, 150))
        self._screen.blit(surf1, (8, sh - _STATUS_BAR_H + 3))
        self._screen.blit(surf2, (8, sh - _STATUS_BAR_H + 25))


# ── 폰트 로드 ────────────────────────────────────────────────────────────────

def _load_fonts() -> dict:
    """OS별 한글 폰트 파일을 직접 찾아 로드한다."""
    import os, sys

    # 실제 폰트 파일 경로 후보 (OS별)
    candidates = [
        # Windows — 맑은 고딕
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        # macOS — Apple SD Gothic / Nanum
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/Library/Fonts/NanumGothic.ttf",
        # Linux
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]

    font_path = None
    for p in candidates:
        if os.path.exists(p):
            font_path = p
            break

    def mk(size: int) -> pygame.font.Font:
        if font_path:
            try:
                return pygame.font.Font(font_path, size)
            except Exception:
                pass
        return pygame.font.Font(None, size + 4)  # 폴백: 영문 비트맵 폰트

    return {
        "title": mk(22),
        "id":    mk(24),
        "label": mk(20),
        "timer": mk(32),
        "btn":   mk(16),
        "small": mk(14),
    }


