# Paper

This is the convenient top-level home for the final manuscript and its compact public evidence bundle.

Files:
- `main.pdf`: compiled paper PDF
- `main.tex`: editable LaTeX source
- `build.ps1`: local build script
- `figures/`: manuscript figure assets copied from the finalized official run
- `source_data/`: CSV source data for the paper figures
- `evidence/`: compact summary files that support the final claim framing

To rebuild:

```powershell
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1
```

Provenance:
- Source run bundle: `results/runs/offv1__featint__platectrl__thrA2_thrR2__z1p96__beta1p0/`
- The manuscript text and figures in this folder are copied from that finalized run output so the paper is easy to find at the repo root.
