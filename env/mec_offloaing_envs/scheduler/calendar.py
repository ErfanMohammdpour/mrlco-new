"""Single-capacity non-preemptive resource calendars with gap search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ResourceCalendar:
    """Earliest non-overlapping interval reservation for capacity=1 resources."""

    name: str
    intervals: list[tuple[float, float]] = field(default_factory=list)

    def reserve(self, duration: float, earliest: float = 0.0) -> tuple[float, float]:
        if duration < 0:
            raise ValueError(f"{self.name}: negative duration")
        if duration == 0.0:
            t = max(0.0, float(earliest))
            return t, t

        earliest = max(0.0, float(earliest))
        duration = float(duration)
        ordered = sorted(self.intervals)

        candidate = earliest
        for start, end in ordered:
            if candidate + duration <= start + 1e-15:
                break
            if end > candidate:
                candidate = end

        reserved = (candidate, candidate + duration)
        self.intervals.append(reserved)
        self.intervals.sort()
        self._assert_no_overlap()
        return reserved

    def _assert_no_overlap(self) -> None:
        ordered = sorted(self.intervals)
        for i in range(1, len(ordered)):
            prev_end = ordered[i - 1][1]
            cur_start = ordered[i][0]
            if cur_start + 1e-12 < prev_end:
                raise RuntimeError(
                    f"{self.name}: overlap {ordered[i - 1]} vs {ordered[i]}"
                )


RESOURCE_NAMES = (
    "UE_CPU",
    "MEC_UL",
    "MEC_CPU",
    "MEC_DL",
    "HELPER_CPU",
    "V2V_CHANNEL",
)


def make_calendars() -> dict[str, ResourceCalendar]:
    return {name: ResourceCalendar(name=name) for name in RESOURCE_NAMES}
