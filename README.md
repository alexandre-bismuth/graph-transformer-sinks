# Why don't Graph Transformers attend to the same token?

Code and results for the NeurIPS 2026 workshop paper *Why don't Graph Transformers attend
to the same token?* (Geometric Distributional Deep Learning workshop, GDDL 2026).

In large language models, attention concentrates on the first token (an *attention sink*).
We find that GRIT, a state-of-the-art Graph Transformer, rarely forms sinks on molecules,
and trace this mainly to its gated attention score: replacing it with a dot product raises
its sink rate from 0.09 to 0.68 on identical graphs.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

ZINC-12k and Peptides-func are downloaded by PyTorch Geometric into `data/` on first use.
The language-model reference (`scripts/lm_sink_reference.py`) additionally needs
`transformers`, `pandas`, `pyarrow` and access to the Hugging Face Hub.

## Reproducing the paper's numbers without a GPU

`outputs/` holds the result files of every run reported in the paper (one directory per
seeded run, `<arm>-s<seed>`). Every number in the paper is regenerated from them by

```bash
python scripts/paper_numbers.py --out-dir paper_numbers
```

which writes `numbers.tex` (one LaTeX macro per number) and the appendix table rows.
Checkpoints (`best_model.pt`) are included only for the GRIT ZINC runs with and without a
virtual node, from which `paper_numbers.py` computes the logit range 5‖a‖₁ (Appendix A) and
the learned virtual-node offset.

## Training and evaluating a model

Each arm is a config in `configs/`, merged over the defaults in `configs/base.yaml`;
GRIT-family arms run with `run_experiment.py`, GraphGPS arms (`zinc_graphgps.yaml`,
`zinc_gps_*.yaml`) with `run_experiment_graphgps.py`. The seed is appended to the run name.
Training writes into `outputs/<run name>` and replaces the shipped results of that run; add
`--logging.save_dir <dir>` to write elsewhere.

```bash
# train (writes outputs/zinc-grit-s0)
python scripts/run_experiment.py --config configs/zinc_grit.yaml --training.seed 0 --no-wandb
python scripts/run_experiment_graphgps.py --config configs/zinc_graphgps.yaml --training.seed 0 --no-wandb

# Peptides-func: full dataset, or one node-count quartile on 2,023 graphs
python scripts/run_experiment.py --config configs/peptides_grit.yaml --training.seed 0 --no-wandb
python scripts/run_experiment.py --config configs/peptides_grit_q1m.yaml --training.seed 0 --no-wandb

# evaluate a trained run (each writes a JSON file into the run directory)
python scripts/run_eval_suite.py --run-dir outputs/zinc-grit-s0            # sink rates, eval_summary.json
python scripts/run_sink_ablation.py --run-dir outputs/zinc-grit-s0         # ablation, renormalised rows
python scripts/run_sink_ablation.py --run-dir outputs/zinc-grit-s0 --no-renorm  # ablation, attention removed
python scripts/run_norm_intervention.py --run-dir outputs/zinc-grit-s0     # Table 4
python scripts/run_symmetry_probe.py --run-dir outputs/zinc-grit-s0        # symmetric-cycle probe
python scripts/run_theory_certificate.py --run-dir outputs/zinc-grit-s0    # bounds of Proposition 3.1
```

`--no-wandb` disables Weights & Biases logging. `scripts/cpu_gate.py` runs quick
correctness checks of the sink metrics and models on CPU.

## Where each result comes from

| Paper | Arms / script |
|---|---|
| Figure 1 | `make_fig1_data.py` (from trained runs), then `plot_fig1.py` (below) |
| Sink rates, ε sweep, peak share (Sec. 2, Tables 1 and 3) | `run_eval_suite.py` |
| Size trend (Sec. 2) | `peptides_grit.yaml`, `peptides_grit_q{1..4}m.yaml` |
| Language models (Sec. 2, App. C) | `lm_sink_reference.py` |
| Bounds, probe, certificate (Sec. 3, App. A–B) | `run_symmetry_probe.py`, `run_theory_certificate.py` |
| Scoring swap and controls (Sec. 4, Table 1) | `zinc_grit_dotlogit.yaml`, `zinc_grit_blind.yaml`, `zinc_grit_dot.yaml`, `zinc_grit_dotbias.yaml`, `zinc_gps_none.yaml`, `zinc_gps_vnode.yaml` |
| Norm intervention (Sec. 4, Table 4) | `run_norm_intervention.py`; `zinc_grit_noclamp.yaml`, `zinc_grit_official.yaml` |
| Virtual nodes (Sec. 4) | `zinc_grit_vnode*.yaml`, `zinc_gps_vnode*.yaml` |
| Ablation ratios (Sec. 4, App. C) | `run_sink_ablation.py` |
| Entropy penalty (Sec. 4) | `zinc_grit_forced_ent{1,2}.yaml` |
| All numbers | `paper_numbers.py` |

Figure 1 (`outputs/figures/fig1_measured.pdf`) is redrawn from the shipped data by

```bash
python scripts/plot_fig1.py outputs/figures/fig1_data.json --panels GRIT GRIT-dotlogit GraphGPS \
  --height 1.5 --subtitles "gated score (published)" "only the logit is a dot product" "GINE + dot product"
```

## Hardware

Each run used a single NVIDIA GH200 GPU. On ZINC, GRIT trains for 2,000 epochs in 8 to 14
hours and GraphGPS for 1,500 epochs in 5.5 to 9 hours; the full Peptides-func model takes
about 7 hours. All analysis scripts also run on CPU.

## Citation

```bibtex
@inproceedings{bismuth2026sinks,
  title     = {Why don't Graph Transformers attend to the same token?},
  author    = {Bismuth, Alexandre},
  booktitle = {NeurIPS 2026 Workshop on Geometric Distributional Deep Learning},
  year      = {2026}
}
```

## Acknowledgements

The GRIT and GraphGPS implementations follow
[GRIT](https://github.com/LiamMa/GRIT) (Ma et al., 2023) and
[GraphGPS](https://github.com/rampasek/GraphGPS) (Rampášek et al., 2022).
