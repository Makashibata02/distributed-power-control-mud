# Pre-publication Review Checklist

No Git repository or remote upload should be created until every required item below is approved.

## Code

- [ ] `src/dpc_mud` contains only the final Part I–4 implementation.
- [ ] Historical branches and exploratory code are absent.
- [ ] The quick reproduction command completes successfully.
- [ ] Fixed seeds and model assumptions are described accurately.

## Configuration and data

- [ ] Both YAML configurations contain publishable parameters only.
- [ ] Published CSV/JSON files contain no raw personal or machine-specific fields.
- [ ] Large raw traces and per-sample Monte Carlo data are intentionally excluded.

## Figures

- [ ] The 23 logical figures in `RESULTS_INDEX.md` are the intended final figures.
- [ ] PNG and SVG render correctly.
- [ ] No figure contains unintended names, identifiers, signatures, or annotations.

## Attribution and license

- [ ] Author name and Fudan University affiliation may remain public.
- [ ] `CITATION.cff` is accurate.
- [ ] MIT License is approved for the included code and generated results.

## Release gate

- [ ] `PRIVACY_AUDIT.md` reports zero blocking findings.
- [ ] `RELEASE_MANIFEST.csv` has been reviewed.
- [ ] Explicit approval has been given before any Git initialization or GitHub upload.
