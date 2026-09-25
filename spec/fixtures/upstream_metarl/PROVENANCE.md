# Vendored upstream scheduler fixture

Source: https://github.com/linkpark/metarl-offloading
Path: `env/mec_offloaing_envs/offloading_env.py`
Commit: `a55094fba33627929b029db2022ce8cccce03649`
sha256: see `SHA256`

This is a byte-identical copy, used only as an AST source for the Part C parity
diagnostic so the comparison is reproducible without network access. It is never
imported or executed (the harness extracts the scheduler function source and
executes that segment). The live clone, when present, takes precedence.
