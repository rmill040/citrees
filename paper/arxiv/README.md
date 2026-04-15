# arXiv Manuscript

- `main.tex` is the manuscript entrypoint.
- `sections/` contains the main-paper content.
- `appendices/` contains proofs and supporting details.

For claims, numbers, and paper structure, do not draft from this README.
Use:

1. `../README.md`
2. `../docs/analysis-contract.md`
3. `../docs/results-finalization.md`
4. `../../STORY.md`

Build:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Live preview:

```bash
latexmk -pdf -pvc -interaction=nonstopmode -halt-on-error main.tex
```
