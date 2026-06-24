# Data Package QA

- Files checked in manifest: 65
- Total manifest payload size: 89.2 MiB
- Files over GitHub 100 MB single-file limit: 0
- SHA256 manifest: `MANIFEST.tsv`
- GitHub repository: https://github.com/wlcbysy/hetao-rse-vpd-salinity-data-code
- Public repository access checked: HTTP 200 on 2026-06-24.
- Release evidence boundary: QC-passed observed GPP is limited to 2021 and 2023; the all-zero 2022 GPP layer is excluded from response modeling.
- GPP unit handling: scaled 8-day MOD17A2H kg C m-2 composites are converted to daily g C m-2 day-1 using a factor of 125.0.
- Model validation included: random held-out validation and 4 x 4 spatial-block cross-validation.
- NIRv triangulation included: annual NIRv comparison with unified exposure metrics, interpreted as an optical check rather than yield-loss validation.
- Included scope: processed current data, annual raster stacks, analysis code, final figures, model-threshold outputs, NIRv triangulation outputs.
- Excluded scope: old backup tables, manuscript DOCX files, rendered page PNGs/PDFs, duplicate release zip, temporary test images, local system files.
- Permanent DOI status: not minted in this local run; repository commit hash should be cited until a Zenodo or OSF DOI is added.
