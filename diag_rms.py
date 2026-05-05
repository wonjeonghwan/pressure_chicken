"""
RMS 진단 스크립트 — FP 원인 분석

정지 딸랑이에서 왜 움직임으로 판정되는지 확인.
매 프레임마다 출력:
  - raw_rms     : bbox 전체 픽셀 기반 RMS (구 방식)
  - masked_rms  : mask 폴리곤 내부 픽셀 기반 RMS (신 방식)
  - mask_px     : mask 내부 유효 픽셀 수
  - centroid_d  : 이전 프레임 대비 mask centroid 이동 거리 (px)
  - crop_d      : 이전 프레임 대비 crop 위치 이동 거리 (px)

사용:
    uv run python diag_rms.py
    uv run python diag_rms.py --burner 1 --frames 150
"""

import argparse
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, ".")

from sources.video_source import VideoSource
from core.detector import BurnerDetector, CLASS_POT_BODY, CLASS_POT_WEIGHT
from core.stabilizer import Stabilizer
import math


def load_config(path="config/store_config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_raw_rms(flow: np.ndarray) -> float:
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.sqrt(np.mean(mag ** 2)))


def compute_masked_rms(flow: np.ndarray, rx1: int, ry1: int,
                       mask_xy: np.ndarray | None) -> tuple[float | None, int]:
    """(masked_rms 구방식 or None, pixel_count) — 비교용으로 유지"""
    if mask_xy is None or len(mask_xy) < 3:
        return None, 0
    rh, rw = flow.shape[:2]
    local_pts = (mask_xy - np.array([[rx1, ry1]], dtype=np.float32)).astype(np.int32)
    mask_img = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillPoly(mask_img, [local_pts], 255)
    px_count = int(np.count_nonzero(mask_img))
    if px_count < 10:
        return None, px_count
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.sqrt(np.mean(mag[mask_img > 0] ** 2))), px_count


def compute_deform_rms(flow: np.ndarray, rx1: int, ry1: int,
                       mask_xy: np.ndarray | None) -> tuple[float | None, int]:
    """(deform_rms 신방식 or None, pixel_count) — mean flow 차감 후 residual RMS"""
    if mask_xy is None or len(mask_xy) < 3:
        return None, 0
    rh, rw = flow.shape[:2]
    local_pts = (mask_xy - np.array([[rx1, ry1]], dtype=np.float32)).astype(np.int32)
    mask_img = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillPoly(mask_img, [local_pts], 255)
    px_count = int(np.count_nonzero(mask_img))
    if px_count < 10:
        return None, px_count
    fx = flow[..., 0][mask_img > 0]
    fy = flow[..., 1][mask_img > 0]
    rx = fx - np.mean(fx)
    ry = fy - np.mean(fy)
    return float(np.sqrt(np.mean(rx ** 2 + ry ** 2))), px_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--burner", type=int, default=None, help="분석할 화구 번호 (없으면 전체)")
    parser.add_argument("--frames", type=int, default=300, help="분석할 프레임 수")
    parser.add_argument("--skip",   type=int, default=0,   help="건너뛸 초기 프레임 수")
    parser.add_argument("--config", default="config/store_config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    burner_ids = (
        [args.burner] if args.burner
        else [b["id"] for b in config["burners"]]
    )
    burner_cfgs = {b["id"]: b for b in config["burners"] if b["id"] in burner_ids}
    if not burner_cfgs:
        print("화구 없음"); return

    src = VideoSource(config["sources"][0])
    src.open()
    stab = Stabilizer(config.get("stabilizer", {}))
    detector = BurnerDetector(
        config["model"]["weights"],
        config["model"]["confidence"]
    )

    fb_params = dict(
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    RMS_THR      = config["optical_flow"]["rms_threshold"]
    NORMALIZE    = config["optical_flow"].get("normalize_rms", False)
    REF_DIAG     = float(config["optical_flow"].get("normalize_ref_diag", 40.0))

    # 화구별 상태
    prev_grays:     dict[int, np.ndarray | None]           = {bid: None for bid in burner_ids}
    prev_centroids: dict[int, tuple[float, float] | None]  = {bid: None for bid in burner_ids}
    prev_crop_boxes:dict[int, tuple[int,int,int,int] | None] = {bid: None for bid in burner_ids}

    # skip
    for _ in range(args.skip):
        ret, frame = src.read()
        if not ret: break

    norm_label = f"norm(÷diag×{REF_DIAG:.0f})" if NORMALIZE else "normalize=OFF"
    print(f"\n[임계값 rms_threshold={RMS_THR}  {norm_label}]  skip={args.skip}프레임\n")
    header = f"{'프레임':>5}  {'화구':>4}  {'bbox_d':>6}  {'raw_rms':>8}  {'deform_rms':>10}  {'norm_rms':>9}  {'mask_px':>7}  {'판정':>8}"
    print(header)
    print("-" * len(header))

    frame_n = 0
    while frame_n < args.frames:
        ret, frame = src.read()
        if not ret or frame is None:
            break
        frame_n += 1

        stabilized = stab.stabilize(frame)
        fh, fw = stabilized.shape[:2]
        margin = 50

        # 전체 ROI를 감싸는 crop 한 번만
        all_rois = [b.get("roi") for b in burner_cfgs.values() if b.get("roi")]
        if not all_rois:
            continue
        min_x = max(0, min(r[0] for r in all_rois) - margin)
        min_y = max(0, min(r[1] for r in all_rois) - margin)
        max_x = min(fw, max(r[0]+r[2] for r in all_rois) + margin)
        max_y = min(fh, max(r[1]+r[3] for r in all_rois) + margin)
        crop_full = stabilized[min_y:max_y, min_x:max_x]

        dets = detector.detect(crop_full)
        for d in dets:
            d.x1 += min_x; d.x2 += min_x
            d.y1 += min_y; d.y2 += min_y
            if d.mask_xy is not None:
                d.mask_xy = d.mask_xy + np.array([[min_x, min_y]], dtype=np.float32)

        bodies  = [d for d in dets if d.class_id == CLASS_POT_BODY   and d.confidence >= 0.3]
        weights = [d for d in dets if d.class_id == CLASS_POT_WEIGHT and d.confidence >= 0.25]

        for bid, bcfg in burner_cfgs.items():
            roi = bcfg.get("roi")
            if not roi:
                continue
            rx_roi, ry_roi, rw_roi, rh_roi = roi

            # 밥솥 매칭
            best_body = None
            mx, my = rw_roi * 0.2, rh_roi * 0.2
            best_dist = float("inf")
            anchor = (rx_roi + rw_roi / 2, ry_roi + rh_roi / 2)
            for b in bodies:
                if (rx_roi - mx <= b.cx <= rx_roi + rw_roi + mx and
                        ry_roi - my <= b.cy <= ry_roi + rh_roi + my):
                    dist = math.hypot(b.cx - anchor[0], b.cy - anchor[1])
                    if dist < best_dist:
                        best_dist, best_body = dist, b

            if best_body is None:
                print(f"{frame_n+args.skip:>5}  {bid:>4}  {'---':>8}  {'---':>10}  {'---':>7}  {'---':>10}  no_body")
                prev_grays[bid] = None
                continue

            # 딸랑이 매칭
            bx1, by1, bx2, by2 = best_body.x1, best_body.y1, best_body.x2, best_body.y2
            bcx = (bx1 + bx2) / 2
            we = (bx2 - bx1) * 0.15
            he = (by2 - by1) * 0.15
            best_weight = None; best_wd = float("inf")
            for w in weights:
                if bx1 - we <= w.cx <= bx2 + we and by1 - he <= w.cy <= by2 + he:
                    d = abs(w.cx - bcx)
                    if d < best_wd:
                        best_wd, best_weight = d, w

            if best_weight is None:
                print(f"{frame_n+args.skip:>5}  {bid:>4}  {'---':>8}  {'---':>10}  {'---':>7}  {'---':>10}  no_weight")
                prev_grays[bid] = None
                continue

            wx1, wy1, wx2, wy2 = best_weight.x1, best_weight.y1, best_weight.x2, best_weight.y2
            mask_xy = best_weight.mask_xy

            # bbox center EMA (optical_flow.py와 동일 로직)
            raw_cx = (wx1 + wx2) / 2.0
            raw_cy = (wy1 + wy2) / 2.0
            pos_alpha = config["optical_flow"].get("pos_ema_alpha", 0.3)
            if prev_centroids[bid] is None:
                ema_cx, ema_cy = raw_cx, raw_cy
            else:
                ema_cx = pos_alpha * raw_cx + (1 - pos_alpha) * prev_centroids[bid][0]
                ema_cy = pos_alpha * raw_cy + (1 - pos_alpha) * prev_centroids[bid][1]

            centroid_d = 0.0
            if prev_centroids[bid] is not None:
                centroid_d = math.hypot(ema_cx - prev_centroids[bid][0],
                                        ema_cy - prev_centroids[bid][1])

            if wy2 <= wy1 or wx2 <= wx1 or wy2 > fh or wx2 > fw:
                prev_grays[bid] = None
                continue

            # EMA centroid 기준으로 crop (optical_flow.py와 동일)
            hw = (wx2 - wx1) // 2
            hh = (wy2 - wy1) // 2
            sx1 = max(0, int(ema_cx) - hw)
            sy1 = max(0, int(ema_cy) - hh)
            sx2 = min(fw, sx1 + 2 * hw)
            sy2 = min(fh, sy1 + 2 * hh)

            roi_gray = cv2.cvtColor(stabilized[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2GRAY)

            raw_rms_val    = 0.0
            deform_rms_val = None
            norm_rms_val   = None
            mask_px        = 0
            verdict        = "skip"
            bbox_d_val     = math.hypot(wx2 - wx1, wy2 - wy1)

            pg = prev_grays[bid]
            if pg is not None and pg.shape == roi_gray.shape:
                flow = cv2.calcOpticalFlowFarneback(pg, roi_gray, None, **fb_params)
                mag_full = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                raw_rms_val = float(np.sqrt(np.mean(mag_full ** 2)))
                deform_rms_val, mask_px = compute_deform_rms(flow, sx1, sy1, mask_xy)
                # fallback: 마스크 없으면 bbox 전체 mean 차감
                if deform_rms_val is None:
                    fx = flow[..., 0].ravel(); fy = flow[..., 1].ravel()
                    rx = fx - np.mean(fx);     ry = fy - np.mean(fy)
                    deform_rms_val = float(np.sqrt(np.mean(rx ** 2 + ry ** 2)))
                if NORMALIZE and bbox_d_val > 0:
                    norm_rms_val = deform_rms_val * REF_DIAG / bbox_d_val
                else:
                    norm_rms_val = deform_rms_val
                cmp_val = norm_rms_val if NORMALIZE else deform_rms_val
                verdict = "MOTION !" if cmp_val > RMS_THR else "still"

            deform_str = f"{deform_rms_val:.3f}" if deform_rms_val is not None else "       ---"
            norm_str   = f"{norm_rms_val:.3f}"   if norm_rms_val   is not None else "      ---"
            print(f"{frame_n+args.skip:>5}  {bid:>4}  {bbox_d_val:>6.1f}  {raw_rms_val:>8.3f}  "
                  f"{deform_str:>10}  {norm_str:>9}  {mask_px:>7}  {verdict}")

            prev_grays[bid]      = roi_gray
            prev_centroids[bid]  = (ema_cx, ema_cy)
            prev_crop_boxes[bid] = (sx1, sy1, sx2, sy2)

    src.release()
    print("\n완료.")


if __name__ == "__main__":
    main()
