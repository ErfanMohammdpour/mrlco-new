# ADR-001: Energy Scope

Status: Accepted for draft spec  
Decision date: 2026-09-03

## Decision

Primary optimized energy metric is `total_mobile_joules`:

- UE compute + UE radio
- HELPER compute + HELPER V2V radio
- excludes MEC compute from primary objective

MEC compute energy MUST be logged separately when modeled.

## Why

This avoids claiming full-system datacenter realism while still charging cooperative helper cost.

## Consequence

- paper MUST say mobile-device-side cooperative energy, not total system energy
- reward/accounting MUST separate UE, helper, and optional MEC components
