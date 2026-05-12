"""
화구별 상태머신 및 타이머 관리

상태 (7개):
  EMPTY               → 빈 화구 (회색)
  POT_IDLE            → 밥솥 감지, 대기 중 (노란색)
  POT_STEAMING_FIRST  → 초벌 타이머 진행 중 (초록색)   ← 타이머 잠금
  DONE_FIRST          → 초벌 완료 냉각 중 (오렌지)      ← 진동 감지 없음, N분 후 자동 전환
  WAIT_SECOND         → 재벌 대기 (파란색)              ← 진동 감지 재활성
  POT_STEAMING_SECOND → 재벌 타이머 진행 중 (진초록)   ← 타이머 잠금
  DONE_SECOND         → 재벌 완료, 경보 (빨간색 점멸)  ← 최종 상태, pot 이탈 시만 EMPTY

전환 규칙:
  EMPTY → POT_IDLE              : pot_body 감지
  POT_IDLE → STEAMING_FIRST     : 딸랑이 진동 확정
  STEAMING_FIRST → DONE_FIRST   : 초벌 타이머 완료
  DONE_FIRST → WAIT_SECOND      : done_first_timeout 경과 + pot 존재
  DONE_FIRST → EMPTY            : pot 이탈
  WAIT_SECOND → STEAMING_SECOND : 딸랑이 진동 확정
  WAIT_SECOND → EMPTY           : pot 이탈
  STEAMING_SECOND → DONE_SECOND : 재벌 타이머 완료
  DONE_SECOND → EMPTY           : pot 이탈 (최종 상태)
"""

from enum import Enum, auto
import time


class BurnerState(Enum):
    EMPTY               = auto()
    POT_IDLE            = auto()
    POT_STEAMING_FIRST  = auto()
    DONE_FIRST          = auto()
    WAIT_SECOND         = auto()
    POT_STEAMING_SECOND = auto()
    DONE_SECOND         = auto()


# 타이머가 진행 중인 상태 집합
_STEAMING = (BurnerState.POT_STEAMING_FIRST, BurnerState.POT_STEAMING_SECOND)

STATE_COLORS = {
    BurnerState.EMPTY:               (80,  80,  80),
    BurnerState.POT_IDLE:            (255, 192,  0),   # 브랜드 옐로우
    BurnerState.POT_STEAMING_FIRST:  (60, 180,  60),
    BurnerState.DONE_FIRST:          (255, 140,  0),   # 오렌지 — 냉각 중
    BurnerState.WAIT_SECOND:         (100, 160, 220),  # 파란색 — 재벌 대기
    BurnerState.POT_STEAMING_SECOND: (30,  130,  30),
    BurnerState.DONE_SECOND:         (220,  40,  40),
}


class BurnerStateMachine:
    """단일 화구의 상태머신 + 카운트다운 타이머"""

    def __init__(
        self,
        burner_id: int,
        countdown_first: int,
        countdown_second: int,
        done_first_timeout: int = 600,
        pot_absent_threshold: int = 60,
    ):
        self.burner_id   = burner_id
        self._cd_first   = countdown_first
        self._cd_second  = countdown_second
        self._done_first_timeout  = done_first_timeout
        self._pot_absent_threshold = pot_absent_threshold  # 완료 상태에서 EMPTY 전환까지 추가 허용 프레임

        self.state: BurnerState           = BurnerState.EMPTY
        self._countdown_end: float | None = None
        self._done_time:      float | None = None
        self._done_first_end: float | None = None  # DONE_FIRST → WAIT_SECOND 전환 시각
        self._pot_absent_count: int = 0             # 연속 미감지 프레임 카운터

        # FrameProcessor가 매 감지마다 갱신
        self.weight_detected: bool  = False
        self.vibration_score: float = 0.0
        self.current_angle:   float | None = None
        self.angle_deviation: float = 0.0

    # ── 외부 이벤트 ────────────────────────────────────────────────────

    def update(self, pot_present: bool, vibrating: bool) -> BurnerState:
        """
        한 프레임의 감지 결과를 받아 상태를 갱신한다.

        STEAMING 상태: 타이머 잠금 — 반드시 완료까지 진행 (수동 초기화만 취소 가능).
        DONE_FIRST  : 진동 감지 없음 — pot 이탈 시 EMPTY, timeout 후 WAIT_SECOND.
        WAIT_SECOND : 진동 감지 재활성 — 재진동 시 재벌 시작, pot 이탈 시 EMPTY.
        DONE_SECOND : 최종 상태 — pot 이탈 시에만 EMPTY.
        """
        if self.state == BurnerState.EMPTY:
            if pot_present:
                self.state = BurnerState.POT_IDLE

        elif self.state == BurnerState.POT_IDLE:
            if not pot_present:
                self.state = BurnerState.EMPTY
            elif vibrating:
                self._start_first()

        elif self.state == BurnerState.POT_STEAMING_FIRST:
            if self.remaining_seconds <= 0:
                self._done_time      = time.monotonic()
                self._done_first_end = time.monotonic() + self._done_first_timeout
                self.state           = BurnerState.DONE_FIRST

        elif self.state == BurnerState.DONE_FIRST:
            # 진동 무시 — pot 이탈(debounce 적용) 또는 timeout 만료만 처리
            if not pot_present:
                self._pot_absent_count += 1
                if self._pot_absent_count >= self._pot_absent_threshold:
                    self._pot_absent_count = 0
                    self.state = BurnerState.EMPTY
            else:
                self._pot_absent_count = 0
                if self._done_first_end is not None and time.monotonic() >= self._done_first_end:
                    self._done_first_end = None
                    self.state = BurnerState.WAIT_SECOND

        elif self.state == BurnerState.WAIT_SECOND:
            if not pot_present:
                self._pot_absent_count += 1
                if self._pot_absent_count >= self._pot_absent_threshold:
                    self._pot_absent_count = 0
                    self.state = BurnerState.EMPTY
            else:
                self._pot_absent_count = 0
                if vibrating:
                    self._start_second()

        elif self.state == BurnerState.POT_STEAMING_SECOND:
            if self.remaining_seconds <= 0:
                self._done_time = time.monotonic()
                self.state      = BurnerState.DONE_SECOND

        elif self.state == BurnerState.DONE_SECOND:
            # 최종 상태 — pot 이탈 시에만 초기화 (debounce 적용)
            if not pot_present:
                self._pot_absent_count += 1
                if self._pot_absent_count >= self._pot_absent_threshold:
                    self._pot_absent_count = 0
                    self.state = BurnerState.EMPTY
            else:
                self._pot_absent_count = 0

        return self.state

    def manual_reset(self) -> None:
        """수동 초기화 → EMPTY"""
        self._reset_timer()
        self.state = BurnerState.EMPTY

    def manual_start(self) -> None:
        """
        수동 타이머 강제 시작/진행.
        EMPTY / POT_IDLE               → 초벌 시작
        POT_STEAMING_FIRST             → 초벌 즉시 완료
        DONE_FIRST                     → 대기 시간 스킵 → WAIT_SECOND
        WAIT_SECOND                    → 재벌 강제 시작
        POT_STEAMING_SECOND            → 재벌 즉시 완료
        """
        if self.state in (BurnerState.EMPTY, BurnerState.POT_IDLE):
            self._start_first()
        elif self.state == BurnerState.POT_STEAMING_FIRST:
            self._countdown_end = time.monotonic()
        elif self.state == BurnerState.DONE_FIRST:
            self._done_first_end = None
            self.state = BurnerState.WAIT_SECOND
        elif self.state == BurnerState.WAIT_SECOND:
            self._start_second()
        elif self.state == BurnerState.POT_STEAMING_SECOND:
            self._countdown_end = time.monotonic()

    # ── 내부 ───────────────────────────────────────────────────────────

    def _start_first(self) -> None:
        self._countdown_end = time.monotonic() + self._cd_first
        self._done_time     = None
        self.state          = BurnerState.POT_STEAMING_FIRST

    def _start_second(self) -> None:
        self._countdown_end = time.monotonic() + self._cd_second
        self._done_time     = None
        self.state          = BurnerState.POT_STEAMING_SECOND

    def _reset_timer(self) -> None:
        self._countdown_end    = None
        self._done_time        = None
        self._done_first_end   = None
        self._pot_absent_count = 0

    # ── 조회 프로퍼티 ──────────────────────────────────────────────────

    @property
    def remaining_seconds(self) -> float:
        end = self._countdown_end
        if self.state not in _STEAMING or end is None:
            return 0.0
        return max(0.0, end - time.monotonic())

    @property
    def remaining_display(self) -> str:
        s = int(self.remaining_seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

    @property
    def phase_label(self) -> str:
        """단계 레이블"""
        return {
            BurnerState.POT_STEAMING_FIRST:  "초벌",
            BurnerState.DONE_FIRST:          "초벌완료",
            BurnerState.WAIT_SECOND:         "재벌대기",
            BurnerState.POT_STEAMING_SECOND: "재벌",
            BurnerState.DONE_SECOND:         "재벌완료",
        }.get(self.state, "")

    @property
    def status_label(self) -> str:
        if self.state == BurnerState.DONE_FIRST and self._done_first_end is not None:
            remaining = max(0.0, self._done_first_end - time.monotonic())
            m, s = divmod(int(remaining), 60)
            return f"{m:02d}:{s:02d}"
        return {
            BurnerState.EMPTY:               "대기",
            BurnerState.POT_IDLE:            "준비",
            BurnerState.POT_STEAMING_FIRST:  self.remaining_display,
            BurnerState.DONE_FIRST:          "초벌완료",
            BurnerState.WAIT_SECOND:         "재벌대기",
            BurnerState.POT_STEAMING_SECOND: self.remaining_display,
            BurnerState.DONE_SECOND:         "완료!",
        }.get(self.state, "대기")

    @property
    def color(self) -> tuple[int, int, int]:
        if self.state == BurnerState.DONE_SECOND:
            elapsed = time.monotonic() - (self._done_time or time.monotonic())
            return STATE_COLORS[BurnerState.DONE_SECOND] if int(elapsed * 2) % 2 == 0 else (80, 10, 10)
        return STATE_COLORS[self.state]

    def __repr__(self) -> str:
        extra = f" [{self.remaining_display}]" if self.state in _STEAMING else ""
        return f"Burner#{self.burner_id}({self.state.name}{extra})"


class BurnerRegistry:
    """매장 전체 화구를 관리하는 컨테이너"""

    def __init__(self) -> None:
        self._burners: dict[int, BurnerStateMachine] = {}

    def add(
        self,
        burner_id: int,
        countdown_first: int,
        countdown_second: int,
        done_first_timeout: int = 600,
        pot_absent_threshold: int = 60,
    ) -> BurnerStateMachine:
        sm = BurnerStateMachine(burner_id, countdown_first, countdown_second, done_first_timeout, pot_absent_threshold)
        self._burners[burner_id] = sm
        return sm

    def get(self, burner_id: int) -> BurnerStateMachine:
        return self._burners[burner_id]

    def all(self) -> list[BurnerStateMachine]:
        return list(self._burners.values())

    def update_all(self, detections: dict[int, tuple[bool, bool]]) -> None:
        """detections: {burner_id: (pot_present, vibrating)}"""
        for burner_id, (pot_present, vibrating) in detections.items():
            if burner_id in self._burners:
                self._burners[burner_id].update(pot_present, vibrating)
