"""
Segmentation 데이터셋 오프라인 증강

YOLO Segmentation 라벨 형식:
  class_id x1 y1 x2 y2 ... xn yn  (정규화 좌표, 가변 꼭짓점)

처리 방식:
  - 폴리곤 꼭짓점 전체를 albumentations keypoints로 변환
  - 기하 변환(flip, affine) 후 꼭짓점 좌표 재조합
  - bbox는 폴리곤 외접 사각형에서 자동 유도 (albumentations 필터링용)
  - 변환 후 화면 밖으로 나간 꼭짓점은 [0, 1]로 클리핑
  - 꼭짓점이 3개 미만 남은 폴리곤은 제거
"""

import os
import cv2
import glob
from pathlib import Path
import albumentations as A


_MAX_PTS = 500  # 폴리곤 꼭짓점 최대 수 (인코딩용 상한, 넘을 일 없음)


def _read_labels(lbl_path: str, iw: int, ih: int):
    """라벨 파싱. bbox(5값)는 4점 사각형 polygon으로 변환하여 형식 통일.

    bbox (pot_body 등):      class cx cy w h         → 4점 rect polygon으로 변환
    polygon (pot_weight 등): class x1 y1 x2 y2 ...  → 그대로 사용

    모든 라벨을 polygon으로 통일 → ultralytics seg 학습 시 빈 마스크 오류 방지.
    """
    rows = []          # 저장용 (polygon 형식으로 통일됨)
    polygons = []      # list of list[(x_pixel, y_pixel)]
    class_labels = []
    bboxes = []        # albumentations yolo bbox (cx, cy, w, h)
    row_indices = []

    if not os.path.exists(lbl_path):
        return rows, polygons, class_labels, bboxes, row_indices

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = list(map(float, parts))
            cls = int(vals[0])
            coords = vals[1:]

            if len(coords) == 4:
                # bbox → 4점 사각형 polygon 변환
                cx, cy, w, h = coords
                x1 = max(0.0, cx - w / 2)
                y1 = max(0.0, cy - h / 2)
                x2 = min(1.0, cx + w / 2)
                y2 = min(1.0, cy + h / 2)
                # 시계방향: top-left, top-right, bottom-right, bottom-left
                poly_norm = [x1, y1, x2, y1, x2, y2, x1, y2]
                poly_px = [(x1*iw, y1*ih), (x2*iw, y1*ih),
                           (x2*iw, y2*ih), (x1*iw, y2*ih)]
                cx_n = (x1 + x2) / 2
                cy_n = (y1 + y2) / 2

            elif len(coords) >= 6:
                # polygon 그대로 사용
                xs_norm = coords[0::2]
                ys_norm = coords[1::2]
                poly_norm = coords
                poly_px = [(x * iw, y * ih) for x, y in zip(xs_norm, ys_norm)]
                x1 = max(0.0, min(xs_norm))
                y1 = max(0.0, min(ys_norm))
                x2 = min(1.0, max(xs_norm))
                y2 = min(1.0, max(ys_norm))
                cx_n = (x1 + x2) / 2
                cy_n = (y1 + y2) / 2

            else:
                continue  # 좌표 부족 → 스킵

            row_idx = len(rows)
            rows.append([cls] + poly_norm)   # polygon 형식으로 통일 저장
            class_labels.append(cls)
            row_indices.append(row_idx)
            bboxes.append([cx_n, cy_n, max(x2 - x1, 1e-4), max(y2 - y1, 1e-4)])
            polygons.append(poly_px)

    return rows, polygons, class_labels, bboxes, row_indices


def _write_labels(path: str, rows: list) -> None:
    with open(path, 'w') as f:
        for row in rows:
            parts = [str(int(row[0]))] + [f"{v:.6f}" for v in row[1:]]
            f.write(" ".join(parts) + "\n")


def _make_compose(extra_transforms: list) -> A.Compose:
    """bbox + keypoint 공통 파라미터로 Compose 생성."""
    return A.Compose(
        extra_transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels', 'row_indices'],
            min_area=0.0,
            min_visibility=0.05,  # 5% 미만 잘린 객체 제거
        ),
        keypoint_params=A.KeypointParams(
            format='xy',
            label_fields=['kp_labels'],
            remove_invisible=False,  # 화면 밖 꼭짓점도 유지 (나중에 클리핑)
        ),
    )


def _flatten_polygons(polygons, row_indices):
    """polygon 행의 꼭짓점을 플랫 리스트로 변환."""
    kp_flat = []
    kp_labels = []
    for poly, row_idx in zip(polygons, row_indices):
        for pt_idx, (x, y) in enumerate(poly):
            kp_flat.append((x, y))
            kp_labels.append(row_idx * _MAX_PTS + pt_idx)
    return kp_flat, kp_labels


def _rebuild_rows(t_bboxes, t_classes, t_row_indices,
                  t_keypoints, t_kp_labels,
                  t_iw, t_ih, orig_poly_lens):
    """변환된 bbox/keypoints → polygon 라벨 행 재조립."""
    surviving = set(t_row_indices)

    # polygon 꼭짓점 복원
    kp_map = {}
    for (kx, ky), encoded in zip(t_keypoints, t_kp_labels):
        row_idx = encoded // _MAX_PTS
        pt_idx = encoded % _MAX_PTS
        if row_idx not in surviving:
            continue
        kx_n = max(0.0, min(1.0, kx / t_iw))
        ky_n = max(0.0, min(1.0, ky / t_ih))
        kp_map.setdefault(row_idx, {})[pt_idx] = (kx_n, ky_n)

    new_rows = []
    for _, cls, row_idx in zip(t_bboxes, t_classes, t_row_indices):
        pts = kp_map.get(row_idx, {})
        if not pts:
            continue
        n_pts = orig_poly_lens.get(row_idx, len(pts))
        coords = []
        for i in range(n_pts):
            if i in pts:
                coords.extend(pts[i])
        if len(coords) < 6:
            continue
        new_rows.append([cls] + coords)
    return new_rows


def _apply_and_save(transform, img, polygons, class_labels, bboxes, row_indices,
                    orig_poly_lens, out_img_path, out_lbl_path) -> bool:
    """transform 적용 후 이미지/라벨 저장. 성공 시 True."""
    try:
        kp_flat, kp_labels = _flatten_polygons(polygons, row_indices)
        result = transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_labels,
            row_indices=row_indices,
            keypoints=kp_flat,
            kp_labels=kp_labels,
        )
        t_img = result['image']
        t_ih, t_iw = t_img.shape[:2]
        new_rows = _rebuild_rows(
            result['bboxes'], result['class_labels'], result['row_indices'],
            result['keypoints'], result['kp_labels'],
            t_iw, t_ih, orig_poly_lens,
        )
        cv2.imwrite(out_img_path, t_img)
        _write_labels(out_lbl_path, new_rows)
        return True
    except Exception as e:
        print(f"증강 실패 ({out_img_path}): {e}")
        return False


def augment_dataset(image_dir, label_dir, output_img_dir, output_lbl_dir, num_augments=26):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # 고정 flip 변환 (항상 적용 — 3가지 조합)
    flip_transforms = [
        _make_compose([A.HorizontalFlip(p=1.0)]),
        _make_compose([A.VerticalFlip(p=1.0)]),
        _make_compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0)]),
    ]
    flip_suffixes = ["_hflip", "_vflip", "_hvflip"]

    # 랜덤 증강 변환 (num_augments 회 반복)
    random_transform = _make_compose([
        A.RandomRotate90(p=0.5),
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)},
            scale=(0.5, 1.5),
            p=0.8,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.3, 0.3),  # 밝기 ±30%
            contrast_limit=(-0.2, 0.2),    # 대비 ±20%
            p=0.7,
        ),
    ])

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"총 {len(image_paths)}장  →  1(원본) + 3(flip) + {num_augments}(랜덤) = {1+3+num_augments}배수  →  예상 {len(image_paths)*(1+3+num_augments)}장")

    for img_path in image_paths:
        base_name = Path(img_path).stem
        lbl_path = os.path.join(label_dir, base_name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            continue
        ih, iw = img.shape[:2]

        rows, polygons, class_labels, bboxes, row_indices = _read_labels(lbl_path, iw, ih)
        orig_poly_lens = {ri: len(poly) for ri, poly in zip(row_indices, polygons)}

        # 1) 원본 저장
        cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_orig.jpg"), img)
        _write_labels(os.path.join(output_lbl_dir, f"{base_name}_orig.txt"), rows)

        # 2) 고정 flip 3종
        for tf, suffix in zip(flip_transforms, flip_suffixes):
            _apply_and_save(
                tf, img, polygons, class_labels, bboxes, row_indices, orig_poly_lens,
                os.path.join(output_img_dir, f"{base_name}{suffix}.jpg"),
                os.path.join(output_lbl_dir, f"{base_name}{suffix}.txt"),
            )

        # 3) 랜덤 증강 num_augments 회
        for i in range(num_augments):
            _apply_and_save(
                random_transform, img, polygons, class_labels, bboxes, row_indices, orig_poly_lens,
                os.path.join(output_img_dir, f"{base_name}_aug_{i}.jpg"),
                os.path.join(output_lbl_dir, f"{base_name}_aug_{i}.txt"),
            )

    print("[완료] 데이터셋 오프라인 증강 완료!")


if __name__ == "__main__":
    print("--- Train 데이터셋 증강 ---")
    augment_dataset(
        "dataset/original/train/images",
        "dataset/original/train/labels",
        "dataset/train_aug/images",
        "dataset/train_aug/labels",
        num_augments=26,
    )

    print("\n--- Validation 데이터셋 증강 ---")
    augment_dataset(
        "dataset/original/valid/images",
        "dataset/original/valid/labels",
        "dataset/valid_aug/images",
        "dataset/valid_aug/labels",
        num_augments=1,
    )
