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

_PAD = 8
_RIGHT_PANEL_W = 400
_RESET_HOLD_S  = 1.0

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
        self._reset_hold:  dict[int, float] = {}

        self._card_rects:  dict[int, pygame.Rect] = {}
        self._reset_rects: dict[int, pygame.Rect] = {}
        self._start_rects: dict[int, pygame.Rect] = {}
        
        # UI 상태
        self.show_mask = True
        self.video_paused = False
        self.dev_mode = False

        # 영상 오버레이 상태
        self._cam_rect = pygame.Rect(0, 0, 0, 0)
        self._cam_scale = 1.0
        self._cam_offset = (0, 0)

        # 캘리브레이션 (설정) 모드
        self.calibration_mode = False
        self._calib_drag_start = None
        self._calib_drag_end = None
        self._calib_dragging = False
        self._calib_burners = []

        # 외부 콜백
        self.on_camera_switch = None

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
        self._selected_id = None
        print("[UI] 캘리브레이션 가이드 모드 시작")

    def save_calibration(self):
        self.calibration_mode = False
        self.config_data["burners"] = self._calib_burners
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
        print(f"[UI] 캘리브레이션 저장 완료 ({len(self._calib_burners)}개 화구)")
        
        # 콜백이 있다면 즉시 반영
        if getattr(self, "on_calibration_saved", None):
            self.on_calibration_saved(self._calib_burners)
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
                self.calibration_mode = False # 원복
            elif key == pygame.K_z:
                if self._calib_burners:
                    self._calib_burners.pop()
            return False

        # 일반 운용 단축키
        if key == pygame.K_m:
            self.show_mask = not self.show_mask
        elif key == pygame.K_c:
            if self.on_camera_switch:
                self.on_camera_switch()
        elif key == pygame.K_ESCAPE:
            self._selected_id = None
        elif key == pygame.K_F2:
            self.start_calibration()
        elif key == pygame.K_d:
            self.dev_mode = not self.dev_mode
            
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
    def _to_video_pos(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        """Pygame 좌표를 카메라 영상 내 실제 픽셀 좌표로 변환"""
        if not self._cam_rect.collidepoint(pos):
            return None
        vx = (pos[0] - self._cam_rect.x) / self._cam_scale
        vy = (pos[1] - self._cam_rect.y) / self._cam_scale
        return (int(vx), int(vy))

    def _on_mouse_down(self, pos: tuple[int, int]) -> None:
        if self.calibration_mode:
            v_pos = self._to_video_pos(pos)
            if v_pos:
                self._calib_dragging = True
                self._calib_drag_start = v_pos
                self._calib_drag_end = v_pos
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

    def _on_mouse_move(self, pos: tuple[int, int]) -> None:
        if self.calibration_mode and self._calib_dragging:
            v_pos = self._to_video_pos(pos)
            if v_pos:
                self._calib_drag_end = v_pos

    def _on_mouse_up(self, pos: tuple[int, int]) -> None:
        if self.calibration_mode and self._calib_dragging:
            self._calib_dragging = False
            v_pos = self._to_video_pos(pos)
            if v_pos and self._calib_drag_start:
                x1, y1 = self._calib_drag_start
                x2, y2 = v_pos
                roi = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
                if roi[2] > 20 and roi[3] > 20:
                    next_id = len(self._calib_burners) + 1
                    self._calib_burners.append({
                        "id": next_id,
                        "source_id": 0,
                        "countdown_first": 720,
                        "countdown_second": 300,
                        "roi": roi
                    })
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
        
        main_w = sw - _RIGHT_PANEL_W
        
        # 1. 왼쪽 카메라 영역 렌더링
        if frames:
            # 지원하는 첫 번째 프레임만 표시
            first_cam = list(frames.values())[0] if len(frames) > 0 else None
            if first_cam is not None:
                self._draw_camera_area(first_cam, processor, main_w, sh)

        if self.calibration_mode:
            self._draw_calibration_overlay(main_w, sh)
        
        # 2. 오른쪽 제어 패널 렌더링
        right_panel = pygame.Rect(main_w, 0, _RIGHT_PANEL_W, sh)
        pygame.draw.rect(self._screen, _C_PANEL, right_panel)
        # 패널 왼쪽 구분선
        pygame.draw.line(self._screen, _C_CARD_BORDER, (main_w, 0), (main_w, sh), 2)
        
        self._draw_right_panel(main_w, sh)

        pygame.display.flip()
        if self._clock:
            self._clock.tick(30) # 30fps UI Redraw

    def _draw_camera_area(self, raw_frame: np.ndarray, processor, box_w: int, box_h: int):
        # 복사본 뷰 객체
        vis = raw_frame.copy()
        
        # 현재 config나 registry 기반 ROI 박스 그리기
        if not self.calibration_mode:
            for bsm in self._registry.all():
                bid = bsm.burner_id
                cfg = next((b for b in self.config_data.get("burners", []) if b["id"] == bid), None)
                if not cfg or "roi" not in cfg: continue
                
                x, y, w, h = cfg["roi"]
                # 흐린 기본 윤곽
                cv2.rectangle(vis, (x, y), (x + w, y + h), (100, 100, 100), 1)

                if processor:
                    r, g, b_val = bsm.color
                    color_bgr = (b_val, g, r)
                    
                    # 밥솥 BBox
                    if bid in processor.last_matched_boxes:
                        bx1, by1, bx2, by2 = processor.last_matched_boxes[bid]
                        
                        # 뷰파인더 스타일 코너 및 얇은 테두리
                        cv2.rectangle(vis, (bx1, by1), (bx2, by2), color_bgr, 1)
                        length = 15
                        for pt1, pt2 in [
                            ((bx1, by1), (bx1+length, by1)), ((bx1, by1), (bx1, by1+length)),
                            ((bx2, by1), (bx2-length, by1)), ((bx2, by1), (bx2, by1+length)),
                            ((bx1, by2), (bx1+length, by2)), ((bx1, by2), (bx1, by2-length)),
                            ((bx2, by2), (bx2-length, by2)), ((bx2, by2), (bx2, by2-length))
                        ]:
                            cv2.line(vis, pt1, pt2, color_bgr, 2)

                        # 라벨 배지 (Badge)
                        text = f"#{bid}"
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(vis, (bx1, by1 - th - 8), (bx1 + tw + 8, by1), color_bgr, -1)
                        # 배경색에 따른 텍스트 색상 (초록 등 어두운 톤이면 흰색, 노랑이면 검정)
                        text_c = (0, 0, 0) if (g > 150 or r > 150) else (255, 255, 255)
                        cv2.putText(vis, text, (bx1 + 4, by1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_c, 2)

                    # 딸랑이 오버레이
                    if bid in processor.last_weight_boxes:
                        wx1, wy1, wx2, wy2 = processor.last_weight_boxes[bid]
                        
                        if self.show_mask and bid in processor.last_mask_xys:
                            pts = processor.last_mask_xys[bid].astype(np.int32)
                            overlay = vis.copy()
                            # 옐로우톤 마스크
                            cv2.fillPoly(overlay, [pts], (0, 200, 255)) 
                            cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
                            cv2.polylines(vis, [pts], True, (0, 180, 255), 1)
                            
                        if self.show_mask:
                            # 얇고 시인성 좋은 흰색(또는 밝은 노랑) 테두리로 딸랑이 박스 표시
                            cv2.rectangle(vis, (wx1, wy1), (wx2, wy2), (255, 255, 255), 1)
                            
                            # 진동 바
                            score = bsm.vibration_score
                            cv2.rectangle(vis, (wx1, wy2+4), (wx2, wy2+10), (60,60,60), -1)
                            fill_w = int((wx2 - wx1) * min(score, 1.0))
                            c = (0, 200, 0) if score < 1.0 else (0, 60, 255)
                            if fill_w > 0:
                                cv2.rectangle(vis, (wx1, wy2+4), (wx1+fill_w, wy2+10), c, -1)
                            
                            if self.dev_mode and bsm.current_angle is not None:
                                score_txt = f"RMS: {bsm.current_angle:.3f}"
                                cv2.putText(vis, score_txt, (wx1, wy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # OpenCV BGR -> Pygame RGB 변환 및 스케일링
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        oh, ow = rgb.shape[:2]
        
        # 화면 비율에 맞추어 축소/확대 최대화
        scale = min(box_w / ow, box_h / oh)
        nw, nh = int(ow * scale), int(oh * scale)
        
        # Pygame Buffer 적용을 위해 shape 맞추고 swap
        rgb = cv2.resize(rgb, (nw, nh))
        
        surf = pygame.image.frombuffer(rgb.tobytes(), (nw, nh), 'RGB')
        
        # Center in box
        cx = (box_w - nw) // 2
        cy = (box_h - nh) // 2
        
        self._cam_rect = pygame.Rect(cx, cy, nw, nh)
        self._cam_scale = scale
        self._screen.blit(surf, (cx, cy))
        
        # (옵션) 비디오 일시정지 오버레이
        if self.video_paused:
            pause_surf = pygame.Surface((nw, nh), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 120))
            txt = self._fonts["title"].render("비디오 일시정지 (Space 재생)", True, _C_BRAND)
            pause_surf.blit(txt, txt.get_rect(center=(nw//2, nh//2)))
            self._screen.blit(pause_surf, (cx, cy))

    def _draw_calibration_overlay(self, box_w, box_h):
        # 캘리브레이션 모드 안내문 및 확정된 화구 Bbox를 Pygame 위에서 그리기
        for b in self._calib_burners:
            rx, ry, rw, rh = b["roi"]
            # 화면상 스케일 적용
            sx = int(rx * self._cam_scale) + self._cam_rect.x
            sy = int(ry * self._cam_scale) + self._cam_rect.y
            sw = int(rw * self._cam_scale)
            sh = int(rh * self._cam_scale)
            
            pygame.draw.rect(self._screen, _C_BRAND, (sx, sy, sw, sh), 2)
            pygame.draw.rect(self._screen, _C_BRAND, (sx, sy, 30, 30))
            txt = self._fonts["id"].render(str(b["id"]), True, _C_TEXT_DARK)
            self._screen.blit(txt, (sx+5, sy+2))
            
        # 드래그 중인 임시 사각형
        if self._calib_dragging and self._calib_drag_start and self._calib_drag_end:
            rx = min(self._calib_drag_start[0], self._calib_drag_end[0])
            ry = min(self._calib_drag_start[1], self._calib_drag_end[1])
            rw = abs(self._calib_drag_end[0] - self._calib_drag_start[0])
            rh = abs(self._calib_drag_end[1] - self._calib_drag_start[1])
            
            sx = int(rx * self._cam_scale) + self._cam_rect.x
            sy = int(ry * self._cam_scale) + self._cam_rect.y
            sw = int(rw * self._cam_scale)
            sh = int(rh * self._cam_scale)
            pygame.draw.rect(self._screen, (255, 100, 100), (sx, sy, sw, sh), 2)
            
        # 상단 안내 바
        banner = pygame.Surface((box_w, 60), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 180))
        self._screen.blit(banner, (0, 0))
        title = self._fonts["title"].render("🛠️ 화구 설정 (Calibration) 모드", True, _C_BRAND)
        desc = self._fonts["small"].render("영상에 드래그하여 화구를 순서대로 그리세요. | Z: 이전 취소 | ENTER: 저장 후 완료", True, _C_TEXT_LIGHT)
        self._screen.blit(title, (20, 10))
        self._screen.blit(desc, (20, 40))

    def _draw_right_panel(self, px: int, ph: int):
        # 상단 헤더 (로고/타이틀 영역)
        header_h = 70
        pygame.draw.rect(self._screen, _C_BRAND, (px, 0, _RIGHT_PANEL_W, header_h))
        title_surf = self._fonts["title"].render("길가옆에 압력밥솥 타이머", True, _C_TEXT_DARK)
        self._screen.blit(title_surf, (px + 20, 20))
        
        # 모델 부재 경고
        oy = header_h + 10
        if self._model_missing:
            wrng = pygame.Rect(px + 10, oy, _RIGHT_PANEL_W - 20, 36)
            pygame.draw.rect(self._screen, _C_WARNING_BG, wrng, border_radius=6)
            msg = self._fonts["small"].render("⚠ AI 모델 없음 (수동 구동만 가능)", True, _C_TEXT_LIGHT)
            self._screen.blit(msg, (px + 20, oy + 10))
            oy += 46

        # 상태 및 툴바
        toolbar_rect = pygame.Rect(px + 10, oy, _RIGHT_PANEL_W - 20, 64)
        pygame.draw.rect(self._screen, _C_CARD_BG, toolbar_rect, border_radius=6)
        
        conf_btn = self._fonts["small"].render("⚙ 설정(F2)", True, _C_BRAND if not self.calibration_mode else _C_TEXT_LIGHT)
        mask_btn = self._fonts["small"].render(f"Mask(M): {'ON' if self.show_mask else 'OFF'}", True, _C_BRAND if self.show_mask else _C_TEXT_LIGHT)
        cam_btn = self._fonts["small"].render("카메라 전환(C)", True, _C_TEXT_LIGHT)
        dev_btn = self._fonts["small"].render(f"Dev(D): {'ON' if getattr(self, 'dev_mode', False) else 'OFF'}", True, _C_BRAND if getattr(self, 'dev_mode', False) else _C_TEXT_LIGHT)
        play_btn = self._fonts["small"].render("일시정지(Space)" if not self.video_paused else "재생(Space)", True, _C_BRAND if self.video_paused else _C_TEXT_LIGHT)

        self._screen.blit(conf_btn, (px + 20, oy + 12))
        self._screen.blit(mask_btn, (px + 110, oy + 12))
        self._screen.blit(cam_btn, (px + 210, oy + 12))
        
        self._screen.blit(dev_btn, (px + 20, oy + 36))
        self._screen.blit(play_btn, (px + 110, oy + 36))
            
        oy += 74

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

        # 스크롤 가능한/리스트 카드 영역 (간단히 Grid 2열로 표시)
        burners = sorted(self._registry.all(), key=lambda b: b.burner_id)
        if not burners:
            return

        cols = 2
        card_w = (_RIGHT_PANEL_W - 30) // cols
        card_h = 100
        
        self._card_rects.clear()
        self._start_rects.clear()
        self._reset_rects.clear()

        for i, bsm in enumerate(burners):
            r = i // cols
            c = i % cols
            cx = px + 10 + c * (card_w + 10)
            cy = oy + r * (card_h + 10)
            self._draw_burner_card(bsm, cx, cy, card_w, card_h)

    def _draw_burner_card(self, bsm, x, y, w, h):
        bid = bsm.burner_id
        selected = (bid == self._selected_id)
        
        card_rect = pygame.Rect(x, y, w, h)
        self._card_rects[bid] = card_rect
        pygame.draw.rect(self._screen, bsm.color, card_rect, border_radius=8)
        
        bcolor = _C_SELECTED if selected else _C_CARD_BORDER
        border_w = 4 if selected else 1
        pygame.draw.rect(self._screen, bcolor, card_rect, border_w, border_radius=8)

        # ID Circle
        pygame.draw.circle(self._screen, _C_BRAND, (x + 22, y + 26), 16)
        id_surf = self._fonts["id"].render(str(bid), True, _C_TEXT_DARK)
        self._screen.blit(id_surf, id_surf.get_rect(center=(x + 22, y + 26)))

        # Phase / Timer
        ph_surf = self._fonts["small"].render(bsm.phase_label or "대기 상태", True, _C_TEXT_LIGHT)
        self._screen.blit(ph_surf, (x + 46, y + 10))

        font = self._fonts["timer"] if bsm.state in _STEAMING else self._fonts["label"]
        color = (255, 255, 255)
        # 만약 조리완료면 붉게/깜박이게
        if bsm.state in (BurnerState.DONE_FIRST, BurnerState.DONE_SECOND):
            color = _C_WARNING_BG if (int(time.time() * 2) % 2 == 0) else _C_TEXT_LIGHT
        
        time_surf = font.render(bsm.status_label, True, color)
        self._screen.blit(time_surf, (x + 46, y + 26))

        # 하단 버튼
        from core.state_machine import BurnerState as BS
        if bsm.state in (BS.EMPTY, BS.POT_IDLE): start_label = "조작"
        elif bsm.state == BS.POT_STEAMING_FIRST: start_label = "완료"
        elif bsm.state == BS.DONE_FIRST: start_label = "재벌"
        elif bsm.state == BS.POT_STEAMING_SECOND: start_label = "완료"
        else: start_label = "시작"

        bw = (w - 20) // 2
        bh = 26
        by = y + h - bh - 8

        btn_r = pygame.Rect(x + 6, by, bw, bh)
        btn_s = pygame.Rect(x + 6 + bw + 8, by, bw, bh)
        
        self._reset_rects[bid] = btn_r
        self._start_rects[bid] = btn_s

        hold_prog = min(1.0, (time.monotonic() - self._reset_hold[bid]) / _RESET_HOLD_S) if bid in self._reset_hold else 0.0
        
        self._draw_btn(btn_r, "초기화(R)" if selected else "리셋", (80, 80, 90), hold_prog)
        self._draw_btn(btn_s, f"{start_label}(S)" if selected else start_label, _C_SUCCESS)

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
    def mk(size):
        return pygame.font.Font(font_path, size) if font_path else pygame.font.Font(None, size + 4)
    return {
        "title": mk(22), "id": mk(20), "label": mk(18), "timer": mk(26), "btn": mk(14), "small": mk(13)
    }
