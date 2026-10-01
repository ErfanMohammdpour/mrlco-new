# Obi et al., IEEE Open Journal of the Industrial Electronics Society, 2026
Table 4 (hardware): Intel Core i9-12900HX @ 2.30 GHz, 8 logical CPUs pinned;
experiments run under WSL2 on this host.
Table 11 (planning-to-control pipeline, numeric values in text/tables, not plots):
  Frenet, 1 thread            mean 86.83 ms    max 101.55 ms
  Validator, single-threaded  mean  0.05 ms    max   0.51 ms
  Trajectory, single-threaded mean  4.42 us    max  28.76 us
The text states Frenet corresponds to the processing of several planning-related
Autoware nodes and that validator/trajectory are single-threaded, all on the
Table 4 hardware.
Caveat: an empirical characterisation of a research WSL2 setup, NOT
production-vehicle timing and not a general guarantee.
Coverage caveat: this source covers planning/control and a few derived Autoware
nodes; it does not measure heavy perception such as object detection, whose
verified evidence remains GPU-based and ineligible for t_ref.
