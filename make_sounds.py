"""
완료 알림음 합성 스크립트 (오프라인, 외부 음원 불필요)

실행: uv run python make_sounds.py
출력: assets/sounds/warn_30s.wav, assets/sounds/complete.wav

설계 의도 — 시끄러운 주방에서도 또렷하게 들리는 크고 분명한 알림. 초벌/재벌 구분 없이 공통.
  · 완료 30초 전 예고(warn_30s): "삑-삑-삐↑" 2회 비프 + 상승음 — 주의 환기형, 약 0.66초
  · 완료(complete)            : 4음 상승 팡파레(G5→C6→E6→G6) 마지막 길게 — 완료 알림, 약 1.0초
배음(2·3 harmonic)을 섞어 사각파에 가깝게 + sustain 엔벨로프 + 음량 0.9로 소음을 뚫는다.
두 소리는 패턴이 확연히 달라(반복 비프 vs 상승 팡파레) 귀로 즉시 구분된다.
음색/길이를 바꾸려면 아래 NOTES 시퀀스의 (주파수Hz, 지속초)만 조정하면 된다.
"""

from __future__ import annotations

import os
import wave

import numpy as np

SR = 44100  # 샘플레이트


def _tone(freq: float, dur: float, harmonics: tuple = (1.0, 0.5, 0.3)) -> np.ndarray:
    """배음을 더한 톤 + sustain 엔벨로프(빠른 attack, 평탄 유지, 짧은 release).

    순수 사인파는 시끄러운 주방에서 묻히므로 2·3배음을 섞어 더 또렷하게(사각파에 가깝게)
    만들고, 지수 감쇠 대신 sustain 엔벨로프로 음량을 끝까지 유지한다.
    freq=0이면 무음(시퀀스 간격용)."""
    t = np.linspace(0.0, dur, int(SR * dur), endpoint=False)
    sig = np.zeros_like(t)
    for k, amp in enumerate(harmonics, start=1):
        sig += amp * np.sin(2.0 * np.pi * freq * k * t)
    env = np.ones_like(t)
    a_n = max(1, int(SR * 0.005))           # 5ms attack — 클릭 제거
    r_n = max(1, int(SR * 0.03))            # 30ms release — 끝맺음
    env[:a_n] = np.linspace(0.0, 1.0, a_n)
    env[-r_n:] = np.linspace(1.0, 0.0, r_n)
    return sig * env


def _sequence(notes: list[tuple[float, float]]) -> np.ndarray:
    """(주파수, 지속초) 시퀀스를 이어붙인 신호."""
    return np.concatenate([_tone(f, d) for f, d in notes])


def _save(path: str, sig: np.ndarray, amp: float = 0.9) -> None:
    """피크 정규화 후 16bit mono wav로 저장."""
    sig = sig / np.max(np.abs(sig)) * amp
    data = (sig * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    print(f"[sound] 생성: {path}  ({len(data) / SR:.2f}s)")


def main() -> None:
    # 완료 30초 전 예고 — 또렷한 2회 비프 + 상승음 (주의 환기)
    _save("assets/sounds/warn_30s.wav", _sequence([
        (1046.5, 0.14), (0.0, 0.07), (1046.5, 0.14), (0.0, 0.07), (1318.5, 0.24),
    ]))
    # 완료 — 큰 4음 상승 팡파레 (마지막 길게)
    _save("assets/sounds/complete.wav", _sequence([
        (784.0, 0.16), (1046.5, 0.16), (1318.5, 0.16), (1568.0, 0.55),
    ]))


if __name__ == "__main__":
    main()
