# ERRATA - v1 geometry and rate claims

Corrects statements made in earlier round reports and messages. Raw v1 artifacts are NOT
modified; this file supersedes the wrong readings only.

## E1. `P(all-MEC winner)` was reversed in one earlier sentence (my error)

Earlier message said "all-MEC is the panel oracle in 60 % of validation graphs / 65 % of
meta-train graphs". The table (and `corrected_geometry_summary.json`) says:

```
validation_query baseline:  all-MEC winner = 0.400   mixed winner = 0.600
meta_train_sample baseline: all-MEC winner = 0.350   mixed winner = 0.650
```

The earlier *tables* were correct; the prose was inverted. Correct reading: all-MEC wins
as candidate-panel winner in 40 % / 35 % of graphs, mixed candidates in 60 % / 65 %.

## E2. "~16x unit error in transfer rates" is WRONG - retracted (my error)

`rate_provenance.json` shows the scheduler's bits->bytes conversion is exactly 8.0 for all
three links (median over 160 graphs: `ul=8.0 dl=8.0 v2v=8.0`). There is no unit bug.

The real explanation of the 7-11 Mbps vs 100-200 Mbps disagreement is **which rate object
is summarized**:

| object | r_mec_ul | r_mec_dl | r_v2v |
|---|---|---|---|
| historical baseline points in `resource_profiles.yaml` (documented, "NOT verified real-world constants") | 7 Mbps | 7 Mbps | 5 Mbps |
| degraded profile (median, 64 graphs) | 8.16 | 8.12 | 3.64 |
| nominal profile (median, 68 graphs) | 37.27 | 34.64 | 17.56 |
| strong profile (median, 28 graphs) | 213.1 | 180.5 | 100.5 |
| all 160 graphs (median) | 18.49 | 22.46 | 10.59 |

The "11 Mbps / 5.6 Mbps" figures are close to the historical/degraded points; the
scheduler uses the per-graph realized draw, whose median is 1.6-3x higher.

## E3. Single-graph rate quoted as if typical (my error)

The earlier "effective 169 Mbps DL / 126 Mbps V2V" came from graph 0 only (a `strong`
profile graph). Profile-stratified medians in E2 are the correct reference. The
per-graph values (160 rows) are in `rate_provenance.csv`.

## E4. "oracle" wording

Every previous use of "oracle" in the geometry work means the **candidate-panel oracle**
over 5 candidates (`all_UE`, `all_MEC`, `all_HELPER`, MC-aware greedy coordinate descent,
HEFT v2). It is not a global optimum over 3^20 placements, and the reported headroom is
**demonstrated candidate-panel headroom**, not an upper bound. Renamed in
`corrected_geometry_summary.json`; a stronger search is required before any "gap to
optimal" language (planned stage 6).

## E5. `mec_share_N` is a surrogate

All `mec_share_*` rows are **SURROGATE CONTENTION SENSITIVITY** (MEC compute and MEC
radio divided by N / processor sharing). They are not multi-user queueing results; v1
schedules one DAG per env instance. Any claim about `N>1` requires the real shared
scheduler (v2 stage 2).

## E6. Earlier transfer-cost comparison

An earlier round compared the frozen v1 transfer durations (MEC_DL 0.48 ms, V2V 1.04 ms,
cut 1.60 ms medians) against an external estimate of 9.1/17.3/26/52 ms and labelled the
difference a unit error. The measured durations are correct for v1 (bytes / derived
byte-rate); the external estimate assumed ~11 Mbps, i.e. the historical/degraded rate
class (E2). Both numbers are self-consistent given their rate inputs; the comparison
must state which rate object it uses.

## E7. `gateF_report.json` provenance gaps

The committed Gate F report was written before the protocol block existed: it has no
`evaluation_protocol` field and `training_fingerprint_parts.code_dirty = "unknown_no_git"`.
Not a correctness bug, but the artifact does not carry protocol identity. Recorded here
instead of rewriting the report.
