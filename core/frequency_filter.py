from __future__ import annotations

from collections import deque

import numpy as np
import scipy.signal


class FrequencyAnalyzer:
    """Phase 3: frequency-domain verification of weight oscillation.

    Accepts a raw float signal per frame (caller decides what signal to use).
    Applies a real-time bandpass IIR filter and tracks EMA of |filtered|.

    Recommended signal: mean horizontal optical flow within weight ROI
      - Oscillating weight: pixels move right then left → mean_flow_x alternates
        sign at oscillation frequency → large filtered amplitude
      - YOLO bbox jitter: random pixel directions → mean ≈ 0 → near-zero filtered
      - Camera shake: removed upstream by Phase 1 stabilizer
      - Lighting / steam: no horizontal flow → near-zero signal

    Example usage:
        mean_x = float(np.mean(oflow.last_flow[..., 0])) if oflow.last_flow is not None else None
        motion, amp = freq.update(mean_x)
    """

    def __init__(self, freq_cfg: dict, fps: float) -> None:
        self._enabled = freq_cfg.get("enabled", True)
        self._band_low = float(freq_cfg.get("band_low_hz", 1.0))
        self._band_high = float(freq_cfg.get("band_high_hz", 8.0))
        self._amp_thr = float(freq_cfg.get("amplitude_threshold", 0.3))
        self._amp_ema_alpha = float(freq_cfg.get("amp_ema_alpha", 0.3))
        self._window = int(freq_cfg.get("window_frames", 60))
        self._trigger = int(freq_cfg.get("trigger_frames", 20))
        self._fps = float(fps)

        self._sos, self._zi = self._build_filter()
        self._amp_ema: float = 0.0

        # buffers for FFT visualization and calibration
        self._signal_buf: deque[float] = deque(maxlen=self._window)
        self._filtered_buf: deque[float] = deque(maxlen=self._window)
        self._cv_hist: deque[bool] = deque(maxlen=self._window)

        self.last_signal: float = 0.0      # raw input signal (e.g. mean_flow_x)
        self.last_filtered: float = 0.0    # bandpass-filtered signal
        self.last_amplitude: float = 0.0   # EMA of |filtered|

    def _build_filter(self) -> tuple:
        nyq = self._fps / 2.0
        low = max(0.001, self._band_low / nyq)
        high = min(0.999, self._band_high / nyq)
        sos = scipy.signal.butter(4, [low, high], btype="band", output="sos")
        zi = scipy.signal.sosfilt_zi(sos)
        return sos, zi

    def reset(self) -> None:
        self._sos, self._zi = self._build_filter()
        self._amp_ema = 0.0
        self._signal_buf.clear()
        self._filtered_buf.clear()
        self._cv_hist.clear()
        self.last_signal = 0.0
        self.last_filtered = 0.0
        self.last_amplitude = 0.0

    @property
    def score(self) -> float:
        if not self._cv_hist:
            return 0.0
        motion_ratio = min(1.0, sum(self._cv_hist) / self._trigger)
        window_ratio = min(1.0, len(self._cv_hist) / self._window)
        return motion_ratio * window_ratio

    def update(self, signal: float | None) -> tuple[bool, float]:
        """Process one frame of signal.

        Args:
            signal: the measurement value for this frame (e.g. mean horizontal
                    optical flow in px/frame), or None if no detection this frame.

        Returns:
            (motion: bool, amplitude: float)
        """
        if not self._enabled:
            return False, 0.0

        if signal is None:
            # decay EMA while signal is unavailable
            self._amp_ema *= (1.0 - self._amp_ema_alpha)
            self.last_amplitude = self._amp_ema
            if len(self._cv_hist) == self._window:
                self._cv_hist.append(False)
            return self._check(), self.last_amplitude

        self.last_signal = signal
        self._signal_buf.append(signal)

        # apply bandpass filter (real-time, maintains state between calls)
        out, self._zi = scipy.signal.sosfilt(self._sos, [signal], zi=self._zi)
        filtered = float(out[0])
        self.last_filtered = filtered
        self._filtered_buf.append(filtered)

        # EMA of |filtered|: amplitude envelope
        self._amp_ema = self._amp_ema_alpha * abs(filtered) + (1.0 - self._amp_ema_alpha) * self._amp_ema
        self.last_amplitude = self._amp_ema

        motion = self._amp_ema > self._amp_thr
        self._cv_hist.append(motion)

        return self._check(), self._amp_ema

    def _check(self) -> bool:
        if len(self._cv_hist) < self._window:
            return False
        return sum(self._cv_hist) >= self._trigger

    def fft_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (freq_hz, amplitude) of current signal buffer.

        Use this to identify the dominant oscillation frequency and calibrate
        band_low_hz / band_high_hz in config.
        """
        if len(self._signal_buf) < 8:
            return np.array([]), np.array([])
        sig = np.array(self._signal_buf, dtype=np.float64)
        sig -= sig.mean()  # remove DC offset
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / self._fps)
        amps = np.abs(np.fft.rfft(sig)) * 2.0 / n
        return freqs, amps

    @property
    def signal_array(self) -> np.ndarray:
        return np.array(self._signal_buf, dtype=np.float64)

    @property
    def filtered_array(self) -> np.ndarray:
        return np.array(self._filtered_buf, dtype=np.float64)
