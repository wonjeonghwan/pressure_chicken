import os
import cv2
import glob
from pathlib import Path
import albumentations as A

def augment_dataset(image_dir, label_dir, output_img_dir, output_lbl_dir, num_augments=5):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_lbl_dir, exist_ok=True)

    # 1. 아구멘테이션 파이프라인 정의 (사용자 요청: 상하반전, 좌우반전, 90도 회전, 평행이동, 줌인/줌아웃)
    transform = A.Compose([
        A.VerticalFlip(p=0.5),   # 50% 확률로 상하 반전 (거꾸로 매달린 밥솥 대응)
        A.HorizontalFlip(p=0.5), # 50% 확률로 좌우 반전
        A.RandomRotate90(p=0.5), # 50% 확률로 90도 회전 (카메라 앵글 틀어짐 대응)
        A.Affine(
            translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # 평행이동 -15% ~ +15%
            scale=(0.5, 1.5), # 줌아웃(0.5x) ~ 줌인(1.5x)
            p=0.8
        )
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_area=0.0, min_visibility=0.1))

    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    print(f"총 {len(image_paths)}장의 원본 이미지를 오프라인 증강합니다...")

    for img_path in image_paths:
        base_name = Path(img_path).stem
        lbl_path = os.path.join(label_dir, base_name + ".txt")

        # 1. 원본 복사
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        bboxes = []
        class_labels = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, x, y, w, h = map(float, parts)
                        bboxes.append([x, y, w, h])
                        class_labels.append(int(c))
        
        # 원본 그대로 output 폴더에 저장
        cv2.imwrite(os.path.join(output_img_dir, f"{base_name}_orig.jpg"), img)
        with open(os.path.join(output_lbl_dir, f"{base_name}_orig.txt"), 'w') as f:
            for b, c in zip(bboxes, class_labels):
                f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")

        # 2. 지정된 횟수만큼 증강본 생성
        for i in range(num_augments):
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                t_img = transformed['image']
                t_bboxes = transformed['bboxes']
                t_classes = transformed['class_labels']

                out_name = f"{base_name}_aug_{i}"
                cv2.imwrite(os.path.join(output_img_dir, f"{out_name}.jpg"), t_img)
                
                with open(os.path.join(output_lbl_dir, f"{out_name}.txt"), 'w') as f:
                    for b, c in zip(t_bboxes, t_classes):
                        f.write(f"{c} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")
            except Exception as e:
                print(f"증강 실패 ({base_name}): {e}")

    print("✅ 데이터셋 오프라인 증강 완료!")

if __name__ == "__main__":
    # Train 데이터셋 증강
    train_img = "dataset/train/images"
    train_lbl = "dataset/train/labels"
    aug_train_img = "dataset/train_aug/images"
    aug_train_lbl = "dataset/train_aug/labels"
    
    print("--- Train 데이터셋 증강 ---")
    augment_dataset(train_img, train_lbl, aug_train_img, aug_train_lbl, num_augments=5)
    
    # Val 데이터셋 증강 (Val은 원본 보존 중심이 좋으나, 탑다운 뷰 평가를 위해 1배수만 추가)
    val_img = "dataset/valid/images"
    val_lbl = "dataset/valid/labels"
    aug_val_img = "dataset/valid_aug/images"
    aug_val_lbl = "dataset/valid_aug/labels"
    
    print("\n--- Validation 데이터셋 증강 ---")
    augment_dataset(val_img, val_lbl, aug_val_img, aug_val_lbl, num_augments=1)
