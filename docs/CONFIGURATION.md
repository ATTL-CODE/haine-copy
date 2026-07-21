# Configuration & Credentials

HAINE hard-codes **no** secrets. All personal/credentialed inputs are supplied at runtime:

| What | How to provide it | Used by |
|---|---|---|
| **Google Earth Engine project** | `export GEE_PROJECT=...` (or `.env`), then `earthengine authenticate` once | Layer 2 (04, 07, 07b) |
| **Output directory** | `export HAINE_DOWNLOAD_DIR=...` (defaults to `~/Downloads`) | Layer 3 plot/JSON saves |
| **AVISO FES2014 tides** | download locally from AVISO, point the script at the folder | `layer1/09_fes2014_tidal_extraction.py` |
| **ERA5 (CDS API)** | put your key in `~/.cdsapirc` (per ECMWF docs) | ERA5 atmospheric extraction |
| **India-WRIS discharge** | download the Neeleeswaram CSV manually | `haine_stochastic_forcing.py::fit_from_series()` |

> None of these files (`.env`, `~/.cdsapirc`, EE tokens, downloaded data) are committed — they are covered by `.gitignore`. Never commit them.
