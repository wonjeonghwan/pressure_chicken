"""
카메라 미리보기 (밝기/노출 테스트용)

실행:
    uv run python videocapture.py
    uv run python videocapture.py --cam 1
    uv run python videocapture.py --exposure -5
    uv run python videocapture.py --gamma 1.5
    uv run python videocapture.py --exposure -5 --gamma 1.5

조작:
    q / ESC   : 종료
    +/-       : gamma +0.1 / -0.1 실시간 조정
    [ / ]     : exposure -1 / +1 실시간 조정
"""

import argparse
import cv2
import numpy as np


def build_lut(gamma: float) -> np.ndarray:
    return np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam",      type=int,   default=0)
    parser.add_argument("--exposure", type=float, default=None)
    parser.add_argument("--gamma",    type=float, default=1.0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"카메라 {args.cam} 열기 실패")
        return

    gamma    = args.gamma
    exposure = args.exposure

    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        actual = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"exposure 설정: {actual}  (요청: {exposure})")
    else:
        actual = cap.get(cv2.CAP_PROP_EXPOSURE)
        print(f"exposure 현재값: {actual}  (미지정 — 자동)")

    print(f"gamma: {gamma}")
    print("조작: +/- gamma 조정  |  [/] exposure 조정  |  q/ESC 종료")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if gamma != 1.0:
            frame = cv2.LUT(frame, build_lut(gamma))

        exp_now = cap.get(cv2.CAP_PROP_EXPOSURE)
        cv2.putText(frame, f"gamma={gamma:.2f}  exposure={exp_now:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Camera Preview", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('+') or key == ord('='):
            gamma = round(gamma + 0.1, 2)
            print(f"gamma → {gamma}")
        elif key == ord('-'):
            gamma = max(0.1, round(gamma - 0.1, 2))
            print(f"gamma → {gamma}")
        elif key == ord(']'):
            exposure = (exposure or exp_now) + 1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"exposure → {cap.get(cv2.CAP_PROP_EXPOSURE)}")
        elif key == ord('['):
            exposure = (exposure or exp_now) - 1
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
            print(f"exposure → {cap.get(cv2.CAP_PROP_EXPOSURE)}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n최종 설정 — gamma: {gamma}  exposure: {cap.get(cv2.CAP_PROP_EXPOSURE)}")


if __name__ == "__main__":
    main()
