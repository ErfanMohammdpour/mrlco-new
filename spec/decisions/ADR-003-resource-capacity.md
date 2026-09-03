# ADR-003: Resource Capacity Model

Status: Accepted for draft spec  
Decision date: 2026-09-03

## Decision

Use explicit single-capacity non-preemptive resources for:

- `UE_CPU`
- `MEC_UL`
- `MEC_CPU`
- `MEC_DL`
- `HELPER_CPU`
- `V2V_CHANNEL`

`V2V_CHANNEL` is half-duplex and shared globally inside one episode.

## Why

This is simplest model that can express precedence and exclusivity without hidden overlap assumptions.

## Consequence

- interval reservation tests become mandatory
- MEC downlink is no longer implicit/infinite
