# Code

| File | Purpose |
|---|---|
| `pruning_study.ipynb` | Main experiment. Defines `eval_object()` (the measurement harness), `install_strategy()` (random / farthest / DINOv2 BoW dispatcher) and `farthest_indices()` (geodesic FPS). Produces the CSVs in `../results/`. |
| `make_figures.py` | Regenerates Figs. 3–6 from `../results/*.csv`. |
| `make_figures_extra.py` | Regenerates the pipeline diagram and the covering-radius figure. |

The notebooks target Google Colab and assume a working
[FoundationPose](https://github.com/NVlabs/FoundationPose) checkout with its
pretrained refiner/scorer weights, plus the YCB-Video `ref_views_16` subset.
Neither is redistributed here.

Scripts write PNGs to the current directory; move them into `../paper/figures/`
to rebuild the paper.
