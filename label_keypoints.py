"""
딸랑이(pot_weight) 키포인트 라벨링 툴

기존 YOLO detection 라벨이 있는 이미지에서
딸랑이의 상단(top)과 하단(bottom) 두 점을 클릭으로 지정하고
YOLO-pose 형식(.txt)으로 저장한다.

저장 형식 (YOLO-pose):
  class cx cy w h  kp_top_x kp_top_y 2  kp_bot_x kp_bot_y 2
  (모두 0~1 정규화)

사용법:
  uv run python label_keypoints.py
  uv run python label_keypoints.py --images dataset/origianl/train/images --labels dataset/origianl/train/labels

조작:
  좌클릭 1번째  : 현재 딸랑이 상단(top) 지점
  좌클릭 2번째  : 현재 딸랑이 하단(bottom) 지점 → 저장 후 다음 딸랑이로
  우클릭        : 현재 클릭 초기화 (다시 찍기)
  n / →        : 이미지 내 남은 딸랑이 skip 하고 다음 이미지
  p / ←        : 이전 이미지
  d             : 현재 이미지 전체 키포인트 삭제
  q / ESC      : 종료
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 한글 폰트 경로 (Windows 기본 맑은 고딕)
_KR_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
_kr_font_cache: dict = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    if size not in _kr_font_cache:
        try:
            _kr_font_cache[size] = ImageFont.truetype(_KR_FONT_PATH, size)
        except Exception:
            _kr_font_cache[size] = ImageFont.load_default()
    return _kr_font_cache[size]


def put_text_kr(img: np.ndarray, text: str, pos: tuple[int, int],
                font_size: int = 14, color: tuple = (180, 180, 180)) -> np.ndarray:
    """PIL로 한글 텍스트를 BGR numpy 이미지에 렌더링."""
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    r, g, b = color[2], color[1], color[0]  # BGR → RGB
    draw.text(pos, text, font=font, fill=(r, g, b))
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

CLASS_POT_WEIGHT = 2

_clicks: list[tuple[int, int]] = []


def _mouse_cb(event, x, y, flags, param):
    global _clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_clicks) < 2:
            _clicks.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        _clicks.clear()


# ── 라벨 IO ────────────────────────────────────────────────────────────────────

def load_labels(path: Path) -> list[list[float]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if parts:
            rows.append([float(v) for v in parts])
    return rows


def save_labels(path: Path, rows: list[list[float]]) -> None:
    lines = [" ".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def weight_indices(rows: list[list[float]]) -> list[int]:
    return [i for i, r in enumerate(rows) if int(r[0]) == CLASS_POT_WEIGHT]


def has_kp(row: list[float]) -> bool:
    return len(row) >= 9


# ── 시각화 ─────────────────────────────────────────────────────────────────────

def draw_state(vis: np.ndarray, rows: list[list[float]],
               cur_wi: int, clicks: list[tuple[int, int]]) -> None:
    """모든 weight box + 키포인트 표시. 현재 작업 중인 것은 밝게 강조."""
    h, w = vis.shape[:2]
    w_idxs = weight_indices(rows)

    for order, ri in enumerate(w_idxs):
        row = rows[ri]
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)

        is_current = (order == cur_wi)
        box_color  = (0, 255, 255) if is_current else (80, 80, 80)
        thickness  = 2 if is_current else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)

        label = f"W{order+1}" + (" ←" if is_current else "")
        cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        if has_kp(row):
            kp_top = (int(row[5] * w), int(row[6] * h))
            kp_bot = (int(row[8] * w), int(row[9] * h))
            cv2.circle(vis, kp_top, 5, (0, 0, 255), -1)
            cv2.circle(vis, kp_bot, 5, (255, 80, 0), -1)
            cv2.line(vis, kp_top, kp_bot, (0, 255, 128), 1)

    # 다른 클래스 박스 (반투명 회색)
    for row in rows:
        if int(row[0]) == CLASS_POT_WEIGHT:
            continue
        cx, cy, bw, bh = row[1], row[2], row[3], row[4]
        x1 = int((cx - bw / 2) * w);  y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w);  y2 = int((cy + bh / 2) * h)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (50, 50, 50), 1)

    # 현재 찍힌 클릭 점
    labels = ["TOP", "BOT"]
    colors = [(0, 0, 255), (255, 80, 0)]
    for i, (px, py) in enumerate(clicks):
        cv2.circle(vis, (px, py), 6, colors[i], -1)
        cv2.putText(vis, labels[i], (px + 8, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2)
    if len(clicks) == 2:
        cv2.line(vis, clicks[0], clicks[1], (0, 255, 128), 2)


# ── 메인 ───────────────────────────────────────────────────────────────────────

def run(images_dir: Path, labels_dir: Path) -> None:
    global _clicks

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    if not image_paths:
        print(f"[오류] 이미지 없음: {images_dir}")
        sys.exit(1)

    labels_dir.mkdir(parents=True, exist_ok=True)
    print(f"이미지 {len(image_paths)}장 | 라벨 저장: {labels_dir}")

    DISP_MAX = 1200
    win = "Keypoint Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, _mouse_cb)

    img_idx = 0
    while 0 <= img_idx < len(image_paths):
        img_path   = image_paths[img_idx]
        label_path = labels_dir / (img_path.stem + ".txt")
        img        = cv2.imread(str(img_path))
        if img is None:
            img_idx += 1
            continue

        ih, iw = img.shape[:2]
        scale  = min(1.0, DISP_MAX / max(iw, ih))
        dw, dh = int(iw * scale), int(ih * scale)

        rows   = load_labels(label_path)
        w_idxs = weight_indices(rows)
        cur_wi = 0   # 현재 작업 중인 weight 순번 (w_idxs 내 인덱스)
        _clicks.clear()

        while True:
            vis = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_AREA)
            draw_state(vis, rows, cur_wi, _clicks)

            total_w  = len(w_idxs)
            done_cnt = sum(1 for ri in w_idxs if has_kp(rows[ri]))
            if total_w == 0:
                hint = "딸랑이(pot_weight) 라벨 없음 — n으로 skip"
            elif cur_wi >= total_w:
                hint = f"✓ 전체 완료({done_cnt}/{total_w}) — n=다음이미지  p=이전"
            else:
                step = "TOP 클릭" if len(_clicks) == 0 else "BOT 클릭"
                hint = f"W{cur_wi+1}/{total_w}  {step}  |  완료:{done_cnt}/{total_w}  우클릭=초기화  n=skip  d=전체삭제"

            title = f"[{img_idx+1}/{len(image_paths)}] {img_path.name[:50]}  |  {hint}"
            cv2.setWindowTitle(win, title)
            vis = put_text_kr(vis, hint, (6, dh - 22), font_size=14, color=(180, 180, 180))
            cv2.imshow(win, vis)

            key = cv2.waitKey(30) & 0xFF

            # ── 두 점 완성 → 현재 weight에 저장 후 다음 weight ──────────────
            if len(_clicks) == 2:
                if cur_wi < total_w:
                    ri = w_idxs[cur_wi]
                    kp_top_x = _clicks[0][0] / dw
                    kp_top_y = _clicks[0][1] / dh
                    kp_bot_x = _clicks[1][0] / dw
                    kp_bot_y = _clicks[1][1] / dh
                    rows[ri] = rows[ri][:5] + [kp_top_x, kp_top_y, 2.0, kp_bot_x, kp_bot_y, 2.0]
                    save_labels(label_path, rows)
                    print(f"  [저장] {img_path.name}  W{cur_wi+1}  top=({kp_top_x:.3f},{kp_top_y:.3f})  bot=({kp_bot_x:.3f},{kp_bot_y:.3f})")
                    cur_wi += 1
                _clicks.clear()
                continue   # 화면 갱신만, 키 입력 불필요

            # ── 키 입력 ──────────────────────────────────────────────────────
            if key in (ord('q'), 27):
                cv2.destroyAllWindows()
                print("[종료]")
                return

            elif key == ord('n') or key == 110:  # n → 다음 이미지
                _clicks.clear()
                img_idx += 1
                break

            elif key == ord('p'):  # p → 이전 이미지
                _clicks.clear()
                img_idx = max(0, img_idx - 1)
                break

            elif key == ord('d'):  # d → 전체 키포인트 삭제
                for ri in w_idxs:
                    rows[ri] = rows[ri][:5]
                save_labels(label_path, rows)
                cur_wi = 0
                _clicks.clear()
                print(f"  [삭제] {img_path.name} 키포인트 전체 제거")

    cv2.destroyAllWindows()
    print("[완료] 모든 이미지 처리됨")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="딸랑이 키포인트 라벨링 툴")
    parser.add_argument("--images", default="dataset/origianl/train/images")
    parser.add_argument("--labels", default="dataset/origianl/train/labels")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    if not images_dir.exists():
        print(f"[오류] 이미지 경로 없음: {images_dir}")
        sys.exit(1)

    run(images_dir, labels_dir)


if __name__ == "__main__":
    main()
