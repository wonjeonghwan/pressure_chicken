"""
화구별 상태머신 및 타이머 관리

상태 (6개):
  EMPTY               → 빈 화구 (회색)
  POT_IDLE            → 밥솥 감지, 딸랑이 정지 (파란색)
  POT_STEAMING_FIRST  → 초벌 타이머 진행 중 (초록색)   ← 타이머 잠금
  DONE_FIRST          → 초벌 완료, 재벌 대기 (노란색)
  POT_STEAMING_SECOND → 재벌 타이머 진행 중 (진초록)   ← 타이머 잠금
  DONE_SECOND         → 재벌 완료, 경보 (빨간색 점멸)

전환 규칙:
  EMPTY → POT_IDLE             : pot_body 감지
  POT_IDLE → STEAMING_FIRST    : 딸랑이 진동 확정
  STEAMING_FIRST → DONE_FIRST  : 초벌 타이머 완료
  DONE_FIRST → STEAMING_SECOND : 딸랑이 재감지 (재벌 시작)
  STEAMING_SECOND → DONE_SECOND: 재벌 타이머 완료

  어느 상태든 + 밥솥 이탈 → EMPTY
  STEAMING 중 → 진동 자동 전환 없음 (잠금), 밥솥 이탈만 예외
"""

from enum import Enum, auto
import time


class BurnerState(Enum):
    EMPTY               = auto()
    POT_IDLE            = auto()
    POT_STEAMING_FIRST  = auto()
    DONE_FIRST          = auto()
    POT_STEAMING_SECOND = auto()
    DONE_SECOND         = auto()


# 타이머가 진행 중인 상태 집합
_STEAMING = (BurnerState.POT_STEAMING_FIRST, BurnerState.POT_STEAMING_SECOND)

STATE_COLORS = {
    BurnerState.EMPTY:               (80,  80,  80),
    BurnerState.POT_IDLE:            (50, 120, 200),
    BurnerState.POT_STEAMING_FIRST:  (60, 180,  60),
    BurnerState.DONE_FIRST:          (200, 160,  20),
    BurnerState.POT_STEAMING_SECOND: (30,  130,  30),
    BurnerState.DONE_SECOND:         (220,  40,  40),
}


class BurnerStateMachine:
    """단일 화구의 상태머신 + 카운트다운 타이머"""

    def __init__(self, burner_id: int, countdown_first: int, countdown_second: int):
        self.burner_id   = burner_id
        self._cd_first   = countdown_first
        self._cd_second  = countdown_second

        self.state: BurnerState           = BurnerState.EMPTY
        self._countdown_end: float | None = None
        self._done_time: float | None     = None

    # ── 외부 이벤트 ────────────────────────────────────────────────────

    def update(self, pot_present: bool, vibrating: bool) -> BurnerState:
        """
        한 프레임의 감지 결과를 받아 상태를 갱신한다.

        STEAMING 상태에서는 진동 결과를 무시하고 타이머만 tick.
        밥솥 이탈(pot_present=False)은 어느 상태에서나 EMPTY 전환.
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
            # 타이머 잠금 — 진동 결과 무시, 밥솥 이탈만 예외
            if not pot_present:
                self._reset_timer()
                self.state = BurnerState.EMPTY
            elif self.remaining_seconds <= 0:
                self._done_time = time.monotonic()
                self.state = BurnerState.DONE_FIRST

        elif self.state == BurnerState.DONE_FIRST:
            if not pot_present:
                self.state = BurnerState.EMPTY
            elif vibrating:
                self._start_second()

        elif self.state == BurnerState.POT_STEAMING_SECOND:
            # 타이머 잠금 — 진동 결과 무시, 밥솥 이탈만 예외
            if not pot_present:
                self._reset_timer()
                self.state = BurnerState.EMPTY
            elif self.remaining_seconds <= 0:
                self._done_time = time.monotonic()
                self.state = BurnerState.DONE_SECOND

        elif self.state == BurnerState.DONE_SECOND:
            if not pot_present:
                self.state = BurnerState.EMPTY

        return self.state

    def manual_reset(self) -> None:
        """수동 초기화 → EMPTY"""
        self._reset_timer()
        self.state = BurnerState.EMPTY

    def manual_start(self) -> None:
        """
        수동 타이머 강제 시작.
        EMPTY / POT_IDLE → 초벌 시작
        DONE_FIRST       → 재벌 시작
        """
        if self.state in (BurnerState.EMPTY, BurnerState.POT_IDLE):
            self._start_first()
        elif self.state == BurnerState.DONE_FIRST:
            self._start_second()

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
        self._countdown_end = None
        self._done_time     = None

    # ── 조회 프로퍼티 ──────────────────────────────────────────────────

    @property
    def remaining_seconds(self) -> float:
        if self.state not in _STEAMING or self._countdown_end is None:
            return 0.0
        return max(0.0, self._countdown_end - time.monotonic())

    @property
    def remaining_display(self) -> str:
        s = int(self.remaining_seconds)
        return f"{s // 60:02d}:{s % 60:02d}"

    @property
    def status_label(self) -> str:
        return {
            BurnerState.EMPTY:               "대기",
            BurnerState.POT_IDLE:            "준비",
            BurnerState.POT_STEAMING_FIRST:  self.remaining_display,
            BurnerState.DONE_FIRST:          "재벌대기",
            BurnerState.POT_STEAMING_SECOND: self.remaining_display,
            BurnerState.DONE_SECOND:         "완료!",
        }[self.state]

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
    ) -> BurnerStateMachine:
        sm = BurnerStateMachine(burner_id, countdown_first, countdown_second)
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
