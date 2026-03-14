import os
import cv2
import glob
from pathlib import Path
import albumentations as A


def _read_labels(lbl_path: str, iw: int, ih: int):
    """라벨 파일 파싱. 키포인트는 픽셀 좌표로 변환해서 반환."""
    rows = []
    bboxes = []
    class_labels = []
    bbox_row_indices = []
    keypoints = []
    kp_labels = []       # row_idx * 2 + kp_slot 로 인코딩된 정수 (albumentations 2.x 호환)
    kp_vis_map = {}      # (row_idx, kp_slot) -> visibility 값

    if not os.path.exists(lbl_path):
        return rows, bboxes, class_labels, bbox_row_indices, keypoints, kp_labels, kp_vis_map

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            vals = list(map(float, parts))
            row_idx = len(rows)
            rows.append(vals)
            # 부동소수점 오차로 경계값이 살짝 벗어날 수 있으므로 [0,1]로 클램핑
            cx, cy, w, h = vals[1], vals[2], vals[3], vals[4]
            x1 = max(0.0, cx - w / 2)
            y1 = max(0.0, cy - h / 2)
            x2 = min(1.0, cx + w / 2)
            y2 = min(1.0, cy + h / 2)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
            bboxes.append([cx, cy, w, h])
            class_labels.append(int(vals[0]))
            bbox_row_indices.append(row_idx)

            # 키포인트 포함 행 (YOLO-pose: class cx cy w h kp_top_x kp_top_y vis kp_bot_x kp_bot_y vis)
            if len(vals) >= 11:
                keypoints.append((vals[5] * iw, vals[6] * ih))
                kp_labels.append(row_idx * 2 + 0)   # top
                kp_vis_map[(row_idx, 0)] = vals[7]

                keypoints.append((vals[8] * iw, vals[9] * ih))
                kp_labels.append(row_idx * 2 + 1)   # bot
                kp_vis_map[(row_idx, 1)] = vals[10]

    return rows, bboxes, class_labels, bbox_row_indices, keypoints, kp_labels, kp_vis_map


def _write_labels(path: str, rows: list) -> None:
    with open(path, 'w') as f:
        for row in rows:
            parts = [str(int(row[0]))] + [f"{v:.6f}" for v in row[1:]]
            f.write(" ".join(parts) + "\n")


def _build_new_rows(t_bboxes, t_classes, t_row_indices,
                    t_keypoints, t_kp_labels, kp_vis_map,
                    t_iw: int, t_ih: int) -> list:
    """증강된 bbox + 키포인트를 합쳐서 새 라벨 행 생성."""
    # orig_row_idx → {kp_slot: (x_norm, y_norm)}
    kp_map: dict = {}
    for kp, encoded in zip(t_keypoints, t_kp_labels):
        kp_x, kp_y = kp[0], kp[1]
        orig_row_idx = encoded // 2
        kp_slot = encoded % 2
        kp_map.setdefault(orig_row_idx, {})[kp_slot] = (kp_x / t_iw, kp_y / t_ih)

    new_rows = []
    for (cx, cy, w, h), cls, orig_idx in zip(t_bboxes, t_classes, t_row_indices):
        row = [cls, cx, cy, w, h]
        kps = kp_map.get(orig_idx, {})
        top = kps.get(0)
        bot = kps.get(1)
        if top is not None and bot is not None:
            vis_top = kp_vis_map.get((orig_idx, 0), 2.0)
            vis_bot = kp_vis_map.get((orig_idx, 1), 2.0)
            row += [top[0], top[1], vis_top, bot[0], bot[1], vis_bot]
        new_rows.append(row)
    return new_rows


def _make_compose(extra_transforms: list) -> A.Compose:
    """bbox + keypoint 공통 파라미터로 Compose 생성."""
    return A.Compose(extra_transforms,
        bbox_params=A.BboxParams(
            format='yolo',
            label_fields=['class_labels', 'bbox_row_indices'],
            min_area=0.0,
            min_visibility=0.1,
        ),
        keypoint_params=A.KeypointParams(
            format='xy',
            label_fields=['kp_labels'],
            remove_invisible=False,
        ))


def _apply_and_save(transform, img, bboxes, class_labels, bbox_row_indices,
                    keypoints, kp_labels, kp_vis_map,
                    out_img_path: str, out_lbl_path: str) -> bool:
    """transform 적용 후 이미지/라벨 저장. 성공 시 True."""
    try:
        result = transform(
            image=img,
            bboxes=bboxes,
            class_labels=class_labels,
            bbox_row_indices=bbox_row_indices,
            keypoints=keypoints,
            kp_labels=kp_labels,
        )
        t_img = result['image']
        t_ih, t_iw = t_img.shape[:2]
        new_rows = _build_new_rows(
            result['bboxes'], result['class_labels'], result['bbox_row_indices'],
            result['keypoints'], result['kp_labels'], kp_vis_map, t_iw, t_ih,
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

    # 고정 flip 변환 (항상 적용 — 4가지 조합)
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

        rows, bboxes, class_labels, bbox_row_indices, keypoints, kp_labels, kp_vis_map = \
            _read_labels(lbl_path, iw, ih)

        # 1) 원본 저장
        cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_orig.jpg"), img)
        _write_labels(os.path.join(output_lbl_dir, f"{base_name}_orig.txt"), rows)

        # 2) 고정 flip 3종
        for tf, suffix in zip(flip_transforms, flip_suffixes):
            _apply_and_save(
                tf, img, bboxes, class_labels, bbox_row_indices, keypoints, kp_labels, kp_vis_map,
                os.path.join(output_img_dir, f"{base_name}{suffix}.jpg"),
                os.path.join(output_lbl_dir, f"{base_name}{suffix}.txt"),
            )

        # 3) 랜덤 증강 num_augments 회
        for i in range(num_augments):
            _apply_and_save(
                random_transform, img, bboxes, class_labels, bbox_row_indices, keypoints, kp_labels, kp_vis_map,
                os.path.join(output_img_dir, f"{base_name}_aug_{i}.jpg"),
                os.path.join(output_lbl_dir, f"{base_name}_aug_{i}.txt"),
            )

    print("[완료] 데이터셋 오프라인 증강 완료!")


if __name__ == "__main__":
    print("--- Train 데이터셋 증강 ---")
    augment_dataset(
        "dataset/origianl/train/images",
        "dataset/origianl/train/labels",
        "dataset/train_aug/images",
        "dataset/train_aug/labels",
        num_augments=26,
    )

    print("\n--- Validation 데이터셋 증강 ---")
    augment_dataset(
        "dataset/origianl/valid/images",
        "dataset/origianl/valid/labels",
        "dataset/valid_aug/images",
        "dataset/valid_aug/labels",
        num_augments=1,
    )
