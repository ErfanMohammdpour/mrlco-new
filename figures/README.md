# MARGO paper figures

## Files referenced by `paper/MARGO_paper_final.tex`

| File | Role |
|------|------|
| `fig_system_model.png` | System model |
| `fig_margo_architecture.png` | Full MARGO pipeline |
| `fig_graph2seq_encoder.png` | Graph2Seq / DAG-edge encoding |
| `fig_triple_readout.png` | Triple readout |
| `fig_decoder_action_generation.png` | LSTM + Luong + ternary actions |
| `fig_meta_reptile_training.png` | PPO + Reptile workflow |
| `fig_joint_objective.png` | Conceptual latency--energy trade-off |
| `fig_graph_fatness_density.png` | Benchmark DAG diversity (fatness / density); 19×100 split described in Sec.~V text |

Export or replace with Nano Banana outputs per `figure_prompts_for_nano_banana.md` (Figs. 1–7; Sec.~V uses `fig_graph_fatness_density.png` instead of separate dataset/pipeline/summary diagrams).

Optional legacy exports from `scripts/export_margo_figures.py` (`system_model.png`, …) are not required by the current LaTeX.
