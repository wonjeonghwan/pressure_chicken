"""
영상/사진에서 학습 프레임 추출 — YOLO 학습 데이터 준비

사용 예:
    # 영상 1개
    uv run python extract_frames.py --video raw/video_001.mp4

    # 영상 여러 개
    uv run python extract_frames.py --video raw/video_001.mp4 raw/video_002.mp4

    # raw/ 폴더 통째로 (영상+사진 자동 구분)
    uv run python extract_frames.py --folder raw

출력: dataset/images/train/ 에 frame_XXXXX.jpg 로 저장
"""

import argparse
import os
import shutil

import cv2

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUT_DIR = "dataset/images/train"


def extract_from_video(video_path: str, fps: float, out_dir: str, start_idx: int) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [건너뜀] 영상 열기 실패: {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval  = max(1, int(video_fps / fps))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  영상: {os.path.basename(video_path)}  ({total}프레임, {video_fps:.1f}fps → {fps}fps 추출)")

    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            out_path = os.path.join(out_dir, f"frame_{start_idx + saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"  → {saved}장 추출")
    return saved


def copy_image(src: str, out_dir: str, idx: int) -> int:
    ext = os.path.splitext(src)[1].lower()
    dst = os.path.join(out_dir, f"frame_{idx:05d}{ext}")
    shutil.copy2(src, dst)
    return 1


def collect_files(folder: str) -> tuple[list[str], list[str]]:
    """폴더에서 영상/이미지 파일 분리"""
    videos, images = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        ext  = os.path.splitext(name)[1].lower()
        if ext in VIDEO_EXTS:
            videos.append(path)
        elif ext in IMAGE_EXTS:
            images.append(path)
    return videos, images


def main() -> None:
    parser = argparse.ArgumentParser(description="영상/사진에서 학습 프레임 추출")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  nargs="+", metavar="FILE", help="영상 파일 (여러 개 가능)")
    group.add_argument("--folder", metavar="DIR",             help="raw 폴더 (영상+사진 자동 처리)")
    parser.add_argument("--fps",   type=float, default=2,     help="영상에서 초당 추출 프레임 수 (기본 2)")
    parser.add_argument("--out",   default=OUT_DIR,           help=f"출력 디렉토리 (기본: {OUT_DIR})")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    total_saved = 0

    if args.video:
        videos = args.video
        images = []
    else:
        if not os.path.isdir(args.folder):
            print(f"[오류] 폴더 없음: {args.folder}")
            return
        videos, images = collect_files(args.folder)
        print(f"[폴더 스캔] {args.folder}")
        print(f"  영상 {len(videos)}개, 사진 {len(images)}개 발견")

    # 영상 처리
    for v in videos:
        total_saved += extract_from_video(v, args.fps, args.out, total_saved)

    # 사진 복사
    if images:
        print(f"  사진 {len(images)}장 복사 중...")
        for img in images:
            total_saved += copy_image(img, args.out, total_saved)
        print(f"  → {len(images)}장 복사 완료")

    print(f"\n[완료] 총 {total_saved}장 → {args.out}/")


if __name__ == "__main__":
    main()
