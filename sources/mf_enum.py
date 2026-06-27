"""
카메라 열거 모듈 (Windows PnP + MF 기반)

PnP LocationInfo : USB 포트 경로 — 같은 포트에 꽂는 한 재부팅·재연결 후에도 동일.
MSMF index       : 런타임 열거 순서 — USB 토폴로지가 바뀌면 달라질 수 있음.

add_camera() 시점에 "미등록 LocationInfo ↔ 미사용 MSMF index" 1:1 매핑을 저장해
각 물리 카메라가 항상 하나의 소스 ID를 유지하도록 한다.
"""
import subprocess
import json
import sys
from dataclasses import dataclass, field


@dataclass
class CameraDevice:
    location: str       # USB 포트 경로 (안정 ID, 예: "0003.0000.0000.007.001...")
    instance_id: str    # PnP InstanceId (예: "USB\\VID_5843&PID_7884&MI_00\\7&21388847&0&0000")
    name: str           # 표시 이름 (예: "2k usb Camera Redeagle")


def list_cameras() -> list[CameraDevice]:
    """
    현재 연결된(Status=OK) 카메라 장치 목록을 LocationInfo 오름차순으로 반환.
    Windows 전용; 다른 플랫폼에서는 빈 목록 반환.
    """
    if sys.platform != "win32":
        return []
    try:
        return _pnp_enumerate()
    except Exception as e:
        print(f"[mf_enum] 카메라 열거 실패: {e}")
        return []


def find_unclaimed(used_locations: set[str]) -> list[CameraDevice]:
    """현재 연결된 카메라 중 config에 등록되지 않은 것만 반환."""
    return [c for c in list_cameras() if c.location not in used_locations]


# ── 내부 구현 ──────────────────────────────────────────────────────────────────

_PS_SCRIPT = """
$cams = Get-PnpDevice -Class Camera | Where-Object Status -eq OK
if ($null -eq $cams) { Write-Output '[]'; exit }
if ($cams -isnot [array]) { $cams = @($cams) }
$result = @()
foreach ($c in $cams) {
    $loc = (Get-PnpDeviceProperty -InstanceId $c.InstanceId `
            -KeyName DEVPKEY_Device_LocationInfo `
            -ErrorAction SilentlyContinue).Data
    $result += [PSCustomObject]@{
        name       = $c.FriendlyName
        instance_id = $c.InstanceId
        location   = if ($loc) { $loc } else { '' }
    }
}
$result | ConvertTo-Json -Depth 2 -Compress
"""


def _pnp_enumerate() -> list[CameraDevice]:
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", _PS_SCRIPT],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0 or not r.stdout.strip():
        return []

    raw = json.loads(r.stdout.strip())
    if isinstance(raw, dict):
        raw = [raw]

    devices = [
        CameraDevice(
            location=item.get("location") or "",
            instance_id=item.get("instance_id") or "",
            name=item.get("name") or "",
        )
        for item in raw
    ]
    # LocationInfo 기준 정렬 → USB 토폴로지 순서 (MSMF 열거 순서와 일치하는 경향)
    devices.sort(key=lambda d: d.location)
    return devices
