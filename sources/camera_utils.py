import json
import cv2
from sources.video_source import VideoSource

def save_config(path: str | None, config: dict) -> None:
    if path is None:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print(f"[camera_utils] config 저장 완료: {path}")
    except Exception as e:
        print(f"[camera_utils] config 저장 실패: {e}")

def switch_camera(sources: dict, source_id: int, current_index: int, 
                  config: dict | None = None, config_path: str | None = None,
                  max_try: int = 10) -> int:
    """카메라 인덱스를 다음으로 순환. 성공 시 새 인덱스 반환."""
    print(f"[camera_utils] 카메라 전환 시도 (현재: {current_index})")
    
    for i in range(1, max_try + 1):
        next_index = (current_index + i) % max_try
        if next_index == current_index:
            continue
            
        print(f"[camera_utils] - 인덱스 {next_index} 시도 중...")
        new_vs = VideoSource({"type": "camera", "index": next_index})
        new_vs.open()
        if not new_vs.failed:
            old_vs = sources.get(source_id)
            if old_vs:
                old_vs.release()
                
            sources[source_id] = new_vs
            print(f"[camera_utils] 카메라 전환 완료: {current_index} → {next_index}")
            
            if config is not None and config_path is not None:
                for sc in config.get("sources", []):
                    if sc.get("id") == source_id:
                        sc["index"] = next_index
                        sc["type"] = "camera"
                save_config(config_path, config)
                
            return next_index
        new_vs.release()

    print(f"[camera_utils] 카메라 전환 실패. 기존 유지: {current_index}")
    return current_index
