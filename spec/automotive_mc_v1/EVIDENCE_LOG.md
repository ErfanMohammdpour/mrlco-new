# EVIDENCE_LOG — primary-source extraction

## 2026-09-30 — 3GPP TS 22.186 V16.2.0 obtained and extracted

**Problem found:** the ETSI delivery path
(`https://www.etsi.org/deliver/etsi_ts/122100_122199/122186/16.02.00_60/ts_122186v160200p.pdf`)
returns **HTTP 403** to a scripted fetch, and the Kish host has no PDF tooling
(`pdftotext`, `pypdf`, `pdfminer` all absent).

**Working path:** the official 3GPP archive
`https://www.3gpp.org/ftp/Specs/archive/22_series/22.186/22186-g20.zip`
(HTTP 200, 96,335 bytes). `g20` is TS 22.186 **V16.2.0** — the exact version cited.
The archive contains `22186-g20.doc` (legacy binary Word, no docx member), which is
readable with the standard library alone:

```
zipfile -> 22186-g20.doc byte stream -> latin-1 decode -> whitespace collapse
```

**Extracted evidence committed:** `evidence/TS122186_v160200_tables_excerpt.txt`
(7,095 bytes, sha256 prefix `8bd6e7f7248939e7`). It contains the flattened row
streams for Tables 5.2-1, 5.3-1, 5.4-1 and 5.5-1, probed by requirement id.
Column order in the flattened stream follows the header: payload (bytes) |
Tx rate (msg/s, absent on some rows) | max end-to-end latency (ms) |
reliability (%) | data rate (Mbps) | min required communication range (m).

**Rows confirmed by direct reading of the extracted stream:**

| requirement | extracted sequence | reading | matches registry row |
|---|---|---|---|
| R.5.3-001 | `2000 100 10 99.99 10` | payload 2000 B, 100 msg/s, latency 10 ms, reliability 99.99%, data rate 10 Mbps | yes |
| R.5.3-006 | `2000 3 99.999 30 500` | payload 2000 B, latency 3 ms, reliability 99.999%, data rate 30 Mbps, range 500 m | yes |
| R.5.2-006 | `50-1200 30 10 99.99 80` | payload 50-1200 B, 30 msg/s, latency 10 ms, reliability 99.99%, data rate 80 Mbps | payload/rate/latency/reliability yes; the 80 Mbps data rate was not previously registered |

These are the values the project owner reported from the standard text; the
independent extraction reproduces them, so the numbers are no longer a
second-hand lead.

**Still open before flags may flip:** per-row column mapping for the Table 5.4-1
and 5.5-1 rows (sensor sharing and remote driving), where the flattened stream
does not carry explicit column labels per row. `transcription_verified` and
`allowed_for_generation` stay **false** for every row until each one has its
`document / section / pdf_page / table / requirement_id / column / value / unit`
recorded and matches the excerpt.

**Verified so far:** document identity, archive URL, requirement ids, and three
complete rows. **Not yet verified:** page numbers (the legacy `.doc` stream has no
reliable page index; the table number and requirement id are used instead, and
this limitation is recorded rather than papered over).
