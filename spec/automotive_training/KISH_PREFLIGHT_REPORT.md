# Kish preflight (Gate 11)

| item | value |
|---|---|
| host | `kish-ai` (185.213.165.199:3031), hostname `aizarepor`, Ubuntu 6.8.0-85, root |
| checkout | `/opt/margo/mrlco-new-6b`, branch `phase4-eval`, HEAD `abe43216`, tree clean before the sync |
| untouchable | `/opt/margo/mrlco-new` — only inspected, never modified |
| sync | `git fetch origin phase4-eval` + `git merge --ff-only` (134a00b → abe4321); later integration files transferred as an LF-normalised tar (a first rsync attempt rewrote 3483 tracked files as CRLF and was fully reverted with `git checkout -- .`) |
| dataset | present and hash-verified against `provenance.json` by the loader |
| step-2 gate | PASS on the frozen checkout |
| environment | Docker images `margo-phase4-tf115-nv2212:latest` (py3.8 + TF 1.15.5 + tf.contrib) and `margo-phase4-tf115-gpu:latest`; `nvidia-container-runtime` present; GPU passthrough verified (`/device:GPU:0`) |
| GPU | single RTX 4090 24564 MiB; **21508 MiB occupied by a vLLM EngineCore (pid 777801, `/opt/pfm-ai`)**; ~2574 MiB free |
| process hygiene | the vLLM is NOT a MARGO process and is not rooted in `-6b`; per the rules it was **not** stopped. No stale MARGO process, no tmux/screen session, no MARGO GPU job was running |
| GPU memory policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`; a 1M-parameter TF1.15 Adam smoke succeeded on the free ~2.5 GiB |
| CUDA/driver/system | untouched; no apt/pip global installs, no reboots, no systemctl |
