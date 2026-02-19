# MARGO Paper - Publication Guide

## 📁 File Structure

```
paper/
├── MARGO_paper_final.tex      # ⭐ MAIN PAPER (Complete, Q1-ready)
├── MARGO_paper_complete.tex   # Previous version
├── MARGO_paper.tex            # Earlier version
├── MARGO_references.bib       # ⭐ BIBLIOGRAPHY (35+ references)
├── EVIDENCE_MAP.md            # ⭐ Code-to-claim mapping
├── FIGURES_GUIDE.md           # ⭐ Figure documentation
├── REPRODUCIBILITY_CHECKLIST.md # Reproducibility info
├── WRITING_GUIDE.md           # Writing guidance (legacy)
├── main.tex                   # Legacy template
├── references.bib             # Legacy references
├── references_updated.bib     # Legacy references
├── equations_and_details.tex  # Mathematical details
├── introduction_revised.tex   # Introduction draft
└── introduction_with_refs.tex # Introduction with citations
```

## 🚀 Quick Start

### To Compile the Paper:

```bash
cd paper
pdflatex MARGO_paper_final.tex
bibtex MARGO_paper_final
pdflatex MARGO_paper_final.tex
pdflatex MARGO_paper_final.tex
```

### Required Files:
1. `MARGO_paper_final.tex` - Main LaTeX source
2. `MARGO_references.bib` - Bibliography
3. `../results/` - All experimental figures

---

## 📊 Experimental Results Summary

### Head-to-Head Comparison: MARGO vs. MRLCO

| Task | Method | Latency | Energy | Δ Latency | Δ Energy |
|------|--------|---------|--------|-----------|----------|
| 1 | MRLCO | 638 ms | 620 J | - | - |
| 1 | **MARGO** | **634 ms** | **595 J** | **-0.63%** | **-4.03%** |
| 2 | MRLCO | 668 ms | 659 J | - | - |
| 2 | **MARGO** | **661 ms** | **633 J** | **-1.05%** | **-3.94%** |

### Comparison with Greedy Baseline

| Task | Greedy Latency | MARGO Latency | Improvement |
|------|----------------|---------------|-------------|
| 1 | 795 ms | 634 ms | **-20.3%** |
| 2 | 810 ms | 661 ms | **-18.4%** |

| Task | Greedy Energy | MARGO Energy | Improvement |
|------|---------------|--------------|-------------|
| 1 | 875 J | 595 J | **-32.0%** |
| 2 | 890 J | 633 J | **-28.9%** |

---

## 📷 Figure Guide

### MRLCO Comparison Figures (`results/mrlco-compare/`)
- `1/task1-latency-with-greedy.jpg` - 3-way comparison
- `1/task1-energy-with-greedy.jpg` - 3-way comparison
- `2/task2-latency-with-greedy.jpg` - 3-way comparison
- `2/task2-energy-with-greedy.jpg` - 3-way comparison

### MARGO Training Dynamics (`results/`)
- `task1-average-latency.jpg` - Latency over 100 iterations
- `task1-average-energy.jpg` - Energy exploration
- `task1-average-reward.jpg` - Reward improvement
- `task1-average-loss.jpg` - Loss convergence
- `task1-value-loses.png` - Value loss
- `task1-policy-loses.jpg` - Policy loss

---

## 🔑 Key Contributions

1. **Graph2Seq Encoder with Triple Readout**
   - GNN-based encoder (2 layers, mean aggregation)
   - Novel triple readout: Attention + Mean + Max pooling
   - Better DAG structure modeling than LSTM

2. **V2V Cooperative Computing**
   - Extended action space: Local/MEC/V2V
   - Half-duplex channel model
   - 40% lower TX power than MEC

3. **Joint Latency-Energy Optimization**
   - Weighted reward function (α=0.5)
   - Comprehensive energy model
   - Pareto-optimal trade-offs

---

## 📚 Key References to Cite

- **MRLCO Baseline**: Wang et al., IEEE TPDS 2020 [wang2020fast]
- **Meta-Learning**: Finn et al., ICML 2017 [finn2017model]
- **Reptile**: Nichol et al., 2018 [nichol2018reptile]
- **PPO**: Schulman et al., 2017 [schulman2017proximal]
- **GraphSAGE**: Hamilton et al., NeurIPS 2017 [hamilton2017inductive]
- **Graph2Seq**: Xu et al., 2018 [xu2018graph2seq]
- **HEFT**: Topcuoglu et al., IEEE TPDS 2002 [topcuoglu2002performance]

---

## ✅ Publication Checklist

- [x] Complete paper structure (Introduction → Conclusion)
- [x] All figures from `results/` included
- [x] MRLCO comparison section with head-to-head results
- [x] V2V + Energy optimization analysis
- [x] Ablation studies (GNN, V2V, Readout, α)
- [x] Comprehensive bibliography (35+ references)
- [x] Mathematical formulations
- [x] Algorithm pseudocode
- [x] Reproducibility details
- [ ] Author information (to be added)
- [ ] Acknowledgments (to be added)
- [ ] Final proofreading

---

## 📝 Paper Statistics

- **Word Count**: ~5,500 words
- **Figures**: 10+ (all from actual experiments)
- **Tables**: 5
- **Equations**: 15+
- **References**: 35+
- **Pages**: ~10-12 (IEEE two-column format)

---

## 🎯 Target Journals (Q1)

1. **IEEE Transactions on Mobile Computing** (TMC) - IF: 7.9
2. **IEEE Transactions on Parallel and Distributed Systems** (TPDS) - IF: 5.6
3. **IEEE Internet of Things Journal** (IoTJ) - IF: 10.6
4. **IEEE Transactions on Vehicular Technology** (TVT) - IF: 6.8
5. **IEEE Access** - IF: 3.9

---

*Last updated: January 2026*

