"""Production scheduling engine package (Phase 1)."""

from .calendar import ResourceCalendar, make_calendars
from .engine import schedule
from .model import (
    CanonicalDAG,
    CanonicalEdge,
    CanonicalTask,
    ConflictingDuplicateEdgeError,
    EnergyBreakdown,
    Location,
    ResourceInterval,
    ScheduleResult,
    TaskExecutionRecord,
    TransferRecord,
)
from .resources import ResourceConfig
from .routes import ROUTE_TABLE, route

__all__ = [
    "CanonicalDAG",
    "CanonicalEdge",
    "CanonicalTask",
    "ConflictingDuplicateEdgeError",
    "EnergyBreakdown",
    "Location",
    "ResourceCalendar",
    "ResourceConfig",
    "ResourceInterval",
    "ROUTE_TABLE",
    "ScheduleResult",
    "TaskExecutionRecord",
    "TransferRecord",
    "make_calendars",
    "route",
    "schedule",
]
