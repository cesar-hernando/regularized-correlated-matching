# Correlated decoding loses the surface code's effective distance below threshold

Code and data for the paper by César Hernando-Fuente, Joonas Majaniemi, Evert van Nieuwenburg and Felix Frohnert
([arXiv:2610.26785](https://arxiv.org/abs/2610.26785)).

Correlated decoding improves matching and clustering surface-code decoders by reweighting one decoding subgraph
conditioned on the other's correction. Under circuit-level noise, a first-pass edge can discount several correlated
edges at once, without accounting for the additional independent faults their joint occurrence requires. Deep below
threshold this lets correlated decoders fail on size-(d−1)/2 fault configurations, a unit loss of effective distance.
Regularizing the reweighting rule, `p'(α) = α P(e_ν | e_μ)` with an error-rate-dependent strength
`α*(p) = α0 / [1 + (p0/p)^k]`, restores the nominal distance scaling and a correlated-decoding advantage.

Headline results (SI1000 noise, rotated surface code, d-round memory):

* Standard correlated matching (CM) loses its advantage as p → 0: at d = 3 it fails about 4.4× as often as plain
  matching at p = 10⁻⁵, and at d = 5 its ratio to matching crosses 1 near p ≈ 4×10⁻⁵.
* Its error-suppression exponents at low p approach (d−1)/2 instead of (d+1)/2 (Table I), and exact fault
  enumeration shows a persistent contribution of size-(d−1)/2 configurations (Fig. 4).
* Regularized correlated matching (RCM) recovers the (d+1)/2 scaling and keeps a constant advantage: at
  p = 10⁻⁴ its ratio to matching is about 0.55 at d = 5 and 0.35 at d = 7.
* The same holds for Riverlane's Local Clustering Decoder (LCD), whose results are included as data; the LCD
  implementation itself is proprietary.

## Installation

The regularized decoder is a [fork of PyMatching](https://github.com/cesar-hernando/PyMatching), installed at its
tag `v2.3.1+rcm.1`, which adds a reweighting strength `alpha` to PyMatching's correlated matching
(`enable_correlations=True`). It also lets `decode`, `decode_batch` and `decode_to_edges_array` take
`edge_reweights`, edges with new weights that apply to that call only. The scripts behind Figs. 7 and 8 use it
to check their own recomputation of correlated matching's posterior weights: plain matching with those weights must
return the same correction as correlated matching's second pass. `make_gallery.py` also uses it to decode each
failure with the reweighting implied by the true error, as if the first pass had been exact.

With [uv](https://docs.astral.sh/uv/), everything, the fork included, installs with

```bash
uv sync
uv run python -c "import pymatching; print(pymatching.__version__)"   # 2.3.1+rcm.1
```

`uv sync` builds the fork from source, so it needs a C++ compiler (Linux: `build-essential`; macOS: Xcode Command
Line Tools; Windows: Visual Studio Build Tools with "Desktop development with C++"); CMake and Ninja are fetched
automatically. On Windows, if the build fails with a path-length error, set `$env:UV_CACHE_DIR = "C:\uvcache"` and run
`uv sync` again.

Without uv, install the fork first (`pip install "pymatching @ git+https://github.com/cesar-hernando/PyMatching@v2.3.1+rcm.1"`),
then `pip install -e .`. Python ≥ 3.11.

Figures are typeset with LaTeX when `latex` and `dvipng` are on the PATH (any TeX Live, MacTeX or MiKTeX install;
the preamble uses `amsmath` and `amssymb`); without them the scripts warn and fall back to matplotlib's mathtext, so
the fonts differ from the paper. Assembling Fig. 8 runs `pdflatex` and does need TeX. Neither LaTeX nor the
compiler can be installed by uv, since they are system programs rather than Python packages.

## Reproducing the paper

All commands run from the repository root (prefix them with `uv run` when using uv). The figures and data shipped
here are their outputs; every command below regenerates them from the data in `data/`.

| Paper | Command | Output |
|---|---|---|
| Analysis tables (α\*, ratios, landscapes) | `python scripts/analyze_rcm_grid_search.py --results-dir data/mwpm/rcm_grid_search_si1000 --tasks tasks/rcm_grid_search_tasks_si1000.txt` | `data/mwpm/rcm_grid_search_si1000/analysis/*.csv` |
| Ratio intervals (Fig. 1, Tables II–III) | `python figures/binomial_ratio_intervals.py data/mwpm/rcm_grid_search_si1000/analysis/ratio_vs_p_curves.csv data/mwpm/rcm_grid_search_si1000/analysis/ratio_vs_p_curves_binomial.csv`<br>`python figures/binomial_ratio_intervals.py data/lcd/ratio_vs_p_curves.csv data/lcd/ratio_vs_p_curves_binomial.csv` | `*_binomial.csv` |
| Fig. 1 | `python figures/si1000_fit_figures.py data/mwpm/rcm_grid_search_si1000/analysis figures/mwpm/si1000 --lcd data/lcd/ratio_vs_p_curves_binomial.csv --lcd-out figures/cm_clcd_ratio_fits.pdf --cm-curves ratio_vs_p_curves_binomial.csv --asymmetric --suffix _v4 --two-panel-only --alpha1-model inverse` | `figures/cm_clcd_ratio_fits_common_named_inside_v4.pdf` |
| Fig. 2 | `python figures/lcd_figures.py landscape figures/mwpm/si1000/cm_landscape_d5_Spectral_r.pdf --distance 5 data/mwpm/rcm_grid_search_si1000/analysis/alpha_landscape_overlay.csv data/mwpm/rcm_grid_search_si1000/analysis/alpha_landscape_surface.csv` | `figures/mwpm/si1000/cm_landscape_d5_Spectral_r.pdf` |
| Fig. 3 | `python figures/make_scaling_csv.py data/mwpm/rcm_grid_search_si1000/*.csv -o data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv`<br>`python figures/lcd_figures.py scaling figures/cm_lcd_scaling.pdf data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv data/lcd/lep_scaling.csv` | `figures/cm_lcd_scaling.pdf` |
| Table I | `python figures/exponents.py data/lcd/lep_scaling.csv data/mwpm/rcm_grid_search_si1000/analysis/lep_scaling.csv` | exponents, as CSV on stdout |
| Figs. 4 and 7 | `python effective_distance_loss/uniform_d7/make_v2.py` | `effective_distance_loss/posterior_and_failing_probability_merged_v2_{contributions,weights}.pdf` |
| Inline d = 3 example | `python effective_distance_loss/failure_gallery/variations.py` | `effective_distance_loss/failure_gallery/fig_d3_reuse_view_above_noalpha_nolegend_nozl.pdf` |
| Fig. 5 | `python figures/xz_memory_ratio.py` | `figures/fig_mc_si1000_xz_ratio.pdf` |
| Fig. 6 | `python scripts/depolarizing_ler_ratio_v4.py` | `figures/mwpm/circuit_level_depolarizing/ler_ratio_fits_v4_ratio.pdf` |
| Tables II, III | `python figures/ratio_fit_table.py data/mwpm/rcm_grid_search_si1000/analysis/ratio_vs_p_curves_binomial.csv data/lcd/ratio_vs_p_curves_binomial.csv --alpha1-model inverse` | LaTeX rows on stdout |
| Fig. 8 | `python effective_distance_loss/failure_gallery/stack_panels.py`<br>`cd effective_distance_loss/failure_gallery && pdflatex fig_mechanisms_stack.tex` | `effective_distance_loss/failure_gallery/fig_mechanisms_stack.pdf` |
| Table IV | `python figures/alpha_star_fit_table.py` | LaTeX rows on stdout |
| Fig. 9 | `python figures/lcd_figures.py landscape figures/mwpm/si1000/cm_landscape_column.pdf --panel-column data/mwpm/rcm_grid_search_si1000/analysis/alpha_landscape_overlay.csv data/mwpm/rcm_grid_search_si1000/analysis/alpha_landscape_surface.csv`<br>`python figures/lcd_figures.py landscape figures/lcd/lcd_landscape_column_gridmin.pdf --panel-column --grid-argmin data/lcd/alpha_landscape_overlay.csv data/lcd/alpha_landscape_surface.csv` | `figures/mwpm/si1000/cm_landscape_column.pdf`, `figures/lcd/lcd_landscape_column_gridmin.pdf` |

The table scripts print fit values with their full uncertainties; the paper rounds them.

## Regenerating the data

These are the expensive steps behind the shipped data; each resumes from its existing output.

* **Grid search (SI1000).** `scripts/run_rcm_grid_search.py` decodes every shot with MWPM and with correlated
  matching at each α of the grid, on the same shots, and writes one CSV per point with the shared-shot counts
  (`only_this_fails`, `only_mwpm_fails`); the paper's error bars do not use them, combining each decoder's
  binomial interval under an independence approximation instead (Appendix D.3). The points are the lines of
  the task lists in `tasks/`:
  `rcm_grid_search_tasks_si1000.txt` (d = 3, 5, 7), `_extension.txt` (d = 5, 7, 9 at low p and d = 9),
  `_topup.txt` (five points with more shots), `_xmem_d5.txt` and `_xmem_d5_deep.txt` (X memory, Fig. 5).
  For example, `python scripts/run_rcm_grid_search.py --tasks tasks/rcm_grid_search_tasks_si1000.txt --indices 0-3 --n-workers 8 --results-dir data/mwpm/rcm_grid_search_si1000`.
  The lowest-p points of d ≥ 7 take up to 1.9×10¹² shots.
* **Grid search (uniform depolarizing, Fig. 6).** `python scripts/run_depolarizing.py --hours H` runs the
  points of `data/mwpm/alpha_star_vs_p.csv` (Stim's generated memory circuits, every noise knob equal to p) at
  α = 1 and at the point's α\*, into `data/mwpm/rcm_grid_search_depolarizing/`. The α\* values come from an α sweep
  over these points, summarized in that file. `scripts/depolarizing_ler_ratio_v4.py` generates the figure and
  leaves out the lowest p of d = 7 and d = 9, where too few logical errors were observed for a reliable ratio.
* **Fault enumeration (Fig. 4).** `python effective_distance_loss/enumerate_faults.py --distances 3 5` enumerates
  every size-(d−1)/2 and size-(d+1)/2 fault set exhaustively (`fault_counts.csv`); `bash effective_distance_loss/uniform_d7/run.sh`
  samples d = 7 uniformly; `effective_distance_loss/uniform_d7/make_v2.py` merges both and draws the figure.
* **Posterior weights (Fig. 7).** `python effective_distance_loss/posterior_weights.py --distances 3 5 7` writes
  `posterior_weights.csv`.
* **Failure examples (Fig. 8).** `python effective_distance_loss/failure_gallery/make_gallery.py` selects the
  examples from the failing sets in `failing_sets/` and writes `examples.json`, which holds every number quoted in Appendix F.

## Repository layout

```
src/RegularizedCM/          decoders, circuits, error decomposition (adapted from Tesseract, see NOTICE), models, statistics
scripts/                    grid-search runner, task-list format, analysis, depolarizing run and figure
tasks/                      grid-search task lists
circuits/si1000_circuits/   SI1000 memory circuits (Z memory d = 3-9, X memory d = 5)
data/mwpm/                  grid-search results: one CSV per point, and analysis/ tables
data/lcd/                   Correlated LCD results (Riverlane)
effective_distance_loss/    fault enumeration, posterior weights and failure examples (Figs. 4, 7, 8)
figures/                    figure and table scripts, and the paper's figures
```

## Citation

If you use this code or data, please cite the paper:

```bibtex
@article{hernandofuente2026correlated,
  title         = {Correlated decoding loses the surface code's effective distance below threshold},
  author        = {Hernando-Fuente, C{\'e}sar and Majaniemi, Joonas and van Nieuwenburg, Evert and Frohnert, Felix},
  journal       = {arXiv preprint arXiv:2610.26785},
  year          = {2026},
  eprint        = {2610.26785},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  doi           = {10.48550/arXiv.2610.26785},
  url           = {https://arxiv.org/abs/2610.26785}
}
```

The same information is in `CITATION.cff`. The error decomposition in `src/RegularizedCM/decompose_errors.py` is
adapted from the Tesseract decoder; please also cite it (see `NOTICE`).

## License

Apache License 2.0, see `LICENSE` and `NOTICE`.
