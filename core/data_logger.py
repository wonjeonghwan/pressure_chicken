"""
감지 로그를 JSONL 파일로 저장.

logs/session_YYYYMMDD_HHMMSS.jsonl 에 두 종류의 이벤트를 기록:
  - state_change : 화구 상태가 바뀔 때마다 즉시 기록
  - snapshot     : 매초 모든 화구 상태 + RMS 값 기록
"""

import json
import os
import time
from datetime import datetime


class DataLogger:
    def __init__(self, log_dir: str = "logs"):
        os.makedirs(log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(log_dir, f"session_{ts}.jsonl")
        self._file = open(self._path, "w", encoding="utf-8", buffering=1)
        self._prev_states: dict[int, str] = {}
        self._last_snapshot = 0.0
        print(f"[logger] 로그 파일: {self._path}")

    def update(self, registry) -> None:
        now = time.time()
        burners = registry.all()

        # 상태 전환 감지
        for bsm in burners:
            bid = bsm.burner_id
            cur = bsm.state.name
            prev = self._prev_states.get(bid)
            if prev != cur:
                self._write({
                    "ts": now,
                    "event": "state_change",
                    "burner_id": bid,
                    "from": prev,
                    "to": cur,
                    "rms": bsm.current_angle,
                    "vibration_score": round(bsm.vibration_score, 4),
                })
                self._prev_states[bid] = cur

        # 매초 스냅샷
        if now - self._last_snapshot >= 1.0:
            self._write({
                "ts": now,
                "event": "snapshot",
                "burners": [
                    {
                        "id": bsm.burner_id,
                        "state": bsm.state.name,
                        "remaining": round(bsm.remaining_seconds, 1),
                        "rms": round(bsm.current_angle, 4) if bsm.current_angle is not None else None,
                        "rms_raw": round(bsm.angle_deviation, 4),
                        "vibration_score": round(bsm.vibration_score, 4),
                        "weight_detected": bsm.weight_detected,
                    }
                    for bsm in burners
                ],
            })
            self._last_snapshot = now

    def _write(self, obj: dict) -> None:
        self._file.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def close(self) -> None:
        self._file.close()
        print(f"[logger] 저장 완료: {self._path}")
