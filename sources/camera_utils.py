import json
import sys
import cv2
from sources.video_source import VideoSource


def save_config(path: str | None, config: dict) -> None:
    if path is None:
        return
    try:
        _path = config.pop("_path", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        if _path is not None:
            config["_path"] = _path
        print(f"[camera_utils] config 저장: {path}")
    except Exception as e:
        print(f"[camera_utils] config 저장 실패: {e}")


def open_camera(index: int) -> VideoSource | None:
    """지정 MSMF index로 카메라를 열고 프레임까지 확인. 실패 시 None."""
    cfg = {"type": "camera", "index": index}
    vs = VideoSource(cfg)
    vs.open()
    if vs.failed:
        return None
    ret, _ = vs.read()
    if not ret:
        vs.release()
        return None
    return vs


def find_unused_index(used_indices: set[int], max_try: int = 10) -> tuple[int, VideoSource] | None:
    """
    used_indices에 없는 MSMF index 중 실제로 프레임이 나오는 첫 번째를 반환.
    성공 시 (index, VideoSource) 반환; 없으면 None.
    """
    for candidate in range(max_try):
        if candidate in used_indices:
            continue
        vs = open_camera(candidate)
        if vs is not None:
            return candidate, vs
    return None


def switch_camera(
    sources: dict,
    source_id: int,
    current_index: int,
    config: dict | None = None,
    config_path: str | None = None,
    max_try: int = 10,
) -> int:
    """
    C 키로 카메라를 다음 사용 가능한 index로 전환.
    성공 시 새 index 반환; 실패 시 current_index 유지.

    참고: 이 함수는 수동 '카메라 전환(C)' 버튼 전용.
          자동 카메라 추가는 main.py add_camera() 참조.
    """
    used = {sc.get("index") for sc in (config or {}).get("sources", [])
            if sc.get("type", "camera") == "camera" and sc.get("id") != source_id}

    print(f"[camera_utils] 카메라 전환 시도 (현재 index={current_index})")
    for i in range(1, max_try + 1):
        next_index = (current_index + i) % max_try
        if next_index in used:
            continue
        vs = open_camera(next_index)
        if vs is None:
            continue

        old_vs = sources.get(source_id)
        if old_vs:
            old_vs.release()
        sources[source_id] = vs

        if config is not None and config_path is not None:
            for sc in config.get("sources", []):
                if sc.get("id") == source_id:
                    sc["index"] = next_index
            save_config(config_path, config)

        print(f"[camera_utils] 전환 완료: {current_index} → {next_index}")
        return next_index

    print(f"[camera_utils] 전환 실패. 현재 index 유지: {current_index}")
    return current_index
