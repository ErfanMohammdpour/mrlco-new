"""3x3 location routing matrix (MARGO-SPEC-v0.1)."""

from __future__ import annotations

from .model import Location

Hop = str

ROUTE_TABLE: dict[tuple[Location, Location], list[Hop]] = {
    (Location.UE, Location.UE): [],
    (Location.UE, Location.MEC): ["MEC_UL"],
    (Location.UE, Location.HELPER): ["V2V"],
    (Location.MEC, Location.UE): ["MEC_DL"],
    (Location.MEC, Location.MEC): [],
    (Location.MEC, Location.HELPER): ["MEC_DL", "V2V"],
    (Location.HELPER, Location.UE): ["V2V"],
    (Location.HELPER, Location.MEC): ["V2V", "MEC_UL"],
    (Location.HELPER, Location.HELPER): [],
}

HOP_TO_RESOURCE = {
    "MEC_UL": "MEC_UL",
    "MEC_DL": "MEC_DL",
    "V2V": "V2V_CHANNEL",
}


def route(src: Location, dst: Location) -> list[Hop]:
    try:
        return list(ROUTE_TABLE[(src, dst)])
    except KeyError as exc:
        raise ValueError(f"no route {src} -> {dst}") from exc


def hop_destination(cur: Location, hop: Hop) -> Location:
    if hop == "MEC_UL":
        return Location.MEC
    if hop == "MEC_DL":
        return Location.UE
    if hop == "V2V":
        if cur == Location.UE:
            return Location.HELPER
        if cur == Location.HELPER:
            return Location.UE
        return Location.HELPER
    raise ValueError(f"unknown hop: {hop}")
