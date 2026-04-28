# Experiment 2 Report

LaTeX source for the relevance-aware DRAM caching report.

## Layout

```
report/
├── main.tex              # 5-page report; uses standard `article` class
├── figures/
│   ├── exp2_heatmaps.png       # 3x3 cache-only contribution plot
│   └── exp2_gap_hitrate.png    # LFU-vs-OPT gap & hit rates
└── README.md             # this file
```

## Building the PDF

### Overleaf (easiest)
1. Zip the `report/` folder.
2. Upload to [Overleaf](https://overleaf.com) -> *New Project* -> *Upload Project*.
3. Set the main document to `main.tex` and click *Recompile*.

### Local pdflatex (TeX Live / MiKTeX)
```
cd report
pdflatex main.tex
pdflatex main.tex     # second pass for cross-references and ToC
```

The document uses only base packages: `geometry`, `graphicx`, `booktabs`, `siunitx`,
`amsmath`, `microtype`, `hyperref`, `xcolor`, `caption`, `subcaption`, `lmodern`.
No external bibliography file (`thebibliography` is inline).

## Regenerating figures

The two PNGs were copied from `../workspace/exp2_heatmaps.png` and
`../workspace/exp2_gap_hitrate.png`, which are produced by running
`workspace/Experiment-2-Caching.ipynb` end-to-end inside the lab Docker container.
After re-running the notebook, refresh this folder with:

```powershell
Copy-Item ..\workspace\exp2_heatmaps.png   figures\exp2_heatmaps.png   -Force
Copy-Item ..\workspace\exp2_gap_hitrate.png figures\exp2_gap_hitrate.png -Force
```
