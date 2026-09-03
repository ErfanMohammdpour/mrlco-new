# ADR-002: Data Residency

Status: Accepted for draft spec  
Decision date: 2026-09-03

## Decision

Task output resides at execution location:

- Local -> `UE`
- MEC -> `MEC`
- V2V -> `HELPER`

Cross-location dependencies MUST move `edge_output_bytes` through explicit routes.

## Why

This removes ambiguity behind V2V -> MEC and MEC -> HELPER precedence.

## Consequence

- no automatic return-to-UE after MEC or HELPER execution
- successor input availability depends on explicit route reservation
