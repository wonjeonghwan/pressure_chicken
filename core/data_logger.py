"""
감지 로그를 JSONL 파일로 저장.

logs/YYYYMMDD.jsonl — 하루 단위 파일, 같은 날 앱 재시작 시 append.

이벤트 종류:
  session_start : 앱 시작 시 1회 기록
  session_end   : 앱 정상 종료 시 기록 (비정상 종료 시 누락)
  state_change  : 화구 상태 전환 즉시 기록
  snapshot      : 매초 모든 화구 상태 + RMS + 카메라 상태 기록

7일 초과 파일은 앱 시작 시 자동 삭제.
"""

import json
import os
import time
from datetime import datetime, timedelta


class DataLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        self._log_dir = log_dir
        self._cleanup_old_logs(keep_days=7)

        now = datetime.now()
        self._session_id  = now.strftime("%Y%m%d_%H%M%S")
        self._path        = os.path.join(log_dir, f"{now.strftime('%Y%m%d')}.jsonl")
        self._file        = open(self._path, "a", encoding="utf-8", buffering=1)
        self._prev_states: dict[int, str] = {}
        self._last_snapshot = 0.0
        self._start_time    = time.time()

        self._write({
            "ts":         self._start_time,
            "event":      "session_start",
            "session_id": self._session_id,
        })
        print(f"[logger] 로그 파일: {self._path}  (세션: {self._session_id})")

    def log_manual_action(self, burner_id: int, action: str, from_state: str) -> None:
        """수동 조작(R키/S키/버튼) 기록."""
        self._write({
            "ts":         time.time(),
            "event":      "manual_action",
            "session_id": self._session_id,
            "burner_id":  burner_id,
            "action":     action,
            "from_state": from_state,
        })

    def log_camera_info(self, sources: dict) -> None:
        """카메라 해상도를 session_start 직후 별도 이벤트로 기록."""
        cam_info = []
        for src_id, vs in sources.items():
            w, h = vs.frame_size
            cam_info.append({"id": src_id, "width": w, "height": h})
        self._write({
            "ts":         time.time(),
            "event":      "camera_info",
            "session_id": self._session_id,
            "cameras":    cam_info,
        })
        for c in cam_info:
            print(f"[logger] Cam-{c['id']}: {c['width']}×{c['height']}")

    def update(self, registry, sources: dict | None = None) -> None:
        now = time.time()
        burners = registry.all()

        # 상태 전환 감지 → 즉시 기록
        for bsm in burners:
            bid  = bsm.burner_id
            cur  = bsm.state.name
            prev = self._prev_states.get(bid)
            if prev != cur:
                self._write({
                    "ts":              now,
                    "event":           "state_change",
                    "session_id":      self._session_id,
                    "burner_id":       bid,
                    "from":            prev,
                    "to":              cur,
                    "rms_smoothed":    round(bsm.current_angle, 4) if bsm.current_angle is not None else None,
                    "vibration_score": round(bsm.vibration_score, 4),
                })
                self._prev_states[bid] = cur

        # 매초 스냅샷
        if now - self._last_snapshot < 1.0:
            return
        self._last_snapshot = now

        cam_status = []
        if sources:
            mono_now = time.monotonic()
            for src_id, vs in sources.items():
                stale_sec = 0.0
                latest_ts = getattr(vs, "_latest_ts", 0.0)
                if latest_ts > 0:
                    stale_sec = round(mono_now - latest_ts, 2)
                cam_status.append({
                    "id":         src_id,
                    "ok":         not vs.failed,
                    "stale_sec":  stale_sec,
                    "fail_count": vs._fail_count,
                })

        self._write({
            "ts":         now,
            "event":      "snapshot",
            "session_id": self._session_id,
            "cameras":    cam_status,
            "burners": [
                {
                    "id":              bsm.burner_id,
                    "state":           bsm.state.name,
                    "remaining":       round(bsm.remaining_seconds, 1),
                    "rms_raw":         round(bsm.raw_rms, 4),
                    "rms_norm":        round(bsm.angle_deviation, 4),
                    "rms_smoothed":    round(bsm.current_angle, 4) if bsm.current_angle is not None else None,
                    "vibration_score": round(bsm.vibration_score, 4),
                    "weight_detected": bsm.weight_detected,
                    "mask_px":         bsm.mask_px,
                }
                for bsm in burners
            ],
        })

    def _write(self, obj: dict) -> None:
        self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def _cleanup_old_logs(self, keep_days: int) -> None:
        cutoff = datetime.now() - timedelta(days=keep_days)
        removed = 0
        for fname in os.listdir(self._log_dir):
            if not fname.endswith(".jsonl"):
                continue
            stem = fname[:-6]  # ".jsonl" 제거
            if len(stem) != 8:
                continue
            try:
                file_date = datetime.strptime(stem, "%Y%m%d")
                if file_date < cutoff:
                    os.remove(os.path.join(self._log_dir, fname))
                    removed += 1
            except ValueError:
                continue
        if removed:
            print(f"[logger] 오래된 로그 {removed}개 삭제 (7일 초과)")

    def close(self) -> None:
        duration = round(time.time() - self._start_time, 1)
        self._write({
            "ts":           time.time(),
            "event":        "session_end",
            "session_id":   self._session_id,
            "duration_sec": duration,
        })
        self._file.close()
        print(f"[logger] 저장 완료: {self._path}  (실행 {duration:.0f}초)")
