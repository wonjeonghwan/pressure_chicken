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

        # STEAMING 중 절대 타이머가 중단되지 않음 (수동 초기화만 가능)
        # 초벌 완료 후 '정지' 구간 관찰 여부 (재벌 대기 조건)
        self._seen_rest_after_first: bool = False

        # FrameProcessor가 매 감지마다 갱신
        self.weight_detected: bool  = False   # 이번 감지에서 pot_weight 있음
        self.vibration_score: float = 0.0     # 진동 진행도 0.0~1.0
        self.current_angle:   float | None = None   # 검은 무게중심 상대 x (px, None=미감지)
        self.angle_deviation: float = 0.0           # 무게중심 x std dev (px)

    # ── 외부 이벤트 ────────────────────────────────────────────────────

    def update(self, pot_present: bool, vibrating: bool) -> BurnerState:
        """
        한 프레임의 감지 결과를 받아 상태를 갱신한다.

        STEAMING 상태에서는 타이머가 항상 완료까지 진행됨 (수동 초기화만 취소 가능).
        POT_IDLE 에서만 밥솥 이탈 시 EMPTY 전환.
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
            # 타이머 잠금 — 한 번 시작하면 반드시 완료까지 진행
            # 밥솥이 가려지거나 인식 안 돼도 절대 취소 안 함
            if self.remaining_seconds <= 0:
                self._done_time = time.monotonic()
                self._seen_rest_after_first = False
                self.state = BurnerState.DONE_FIRST

        elif self.state == BurnerState.DONE_FIRST:
            # 初벌 완료 → 딸랑이 정지 관찰 → 재벌 대기 → 재진동 시 재벌 시작
            if not vibrating:
                self._seen_rest_after_first = True   # 정지 구간 확인
            elif vibrating and self._seen_rest_after_first:
                self._start_second()   # 정지 후 재진동 → 재벌 시작

        elif self.state == BurnerState.POT_STEAMING_SECOND:
            # 타이머 잠금 — 동일하게 반드시 완료까지 진행
            if self.remaining_seconds <= 0:
                self._done_time = time.monotonic()
                self.state = BurnerState.DONE_SECOND

        elif self.state == BurnerState.DONE_SECOND:
            pass  # 수동 초기화 대기 (자동 전환 없음)

        return self.state

    def manual_reset(self) -> None:
        """수동 초기화 → EMPTY"""
        self._reset_timer()
        self._seen_rest_after_first = False
        self.state = BurnerState.EMPTY

    def manual_start(self) -> None:
        """
        수동 타이머 강제 시작/진행.
        EMPTY / POT_IDLE               → 초벌 시작
        POT_STEAMING_FIRST             → 초벌 즉시 완료
        DONE_FIRST                     → 재벌 시작 (강제)
        POT_STEAMING_SECOND            → 재벌 즉시 완료
        """
        if self.state in (BurnerState.EMPTY, BurnerState.POT_IDLE):
            self._start_first()
        elif self.state == BurnerState.POT_STEAMING_FIRST:
            # 즉시 완료
            self._countdown_end = time.monotonic()
        elif self.state == BurnerState.DONE_FIRST:
            self._start_second()
        elif self.state == BurnerState.POT_STEAMING_SECOND:
            # 즉시 완료
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
        self._countdown_end      = None
        self._done_time          = None
        self._seen_rest_after_first = False

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
        """초벌/재벌 단계 레이블 (타이머 진행 중일 때만 표시)"""
        return {
            BurnerState.POT_STEAMING_FIRST:  "초벌",
            BurnerState.DONE_FIRST:          "초벌완료",
            BurnerState.POT_STEAMING_SECOND: "재벌",
            BurnerState.DONE_SECOND:         "재벌완료",
        }.get(self.state, "")

    @property
    def status_label(self) -> str:
        if self.state == BurnerState.DONE_FIRST:
            # 아직 정지 구간을 못 봤으면 "초벌완료", 봤으면 "재벌대기"
            return "재벌대기" if self._seen_rest_after_first else "초벌완료"
        return {
            BurnerState.EMPTY:               "대기",
            BurnerState.POT_IDLE:            "준비",
            BurnerState.POT_STEAMING_FIRST:  self.remaining_display,
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
