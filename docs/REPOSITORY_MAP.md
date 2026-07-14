# Repository map

| Path | Purpose | Tracked? |
|---|---|---|
| `data/labels/` | Final adjudicated AAER candidate decisions | Yes |
| `scripts/data/` | SEC download and dataset construction | Yes |
| `scripts/analysis/` | Main benchmark and robustness analyses | Yes |
| `config/` | Frozen scientific parameters and seeds | Yes |
| `results/` | Compact tables, reports, and hashes | Yes |
| `figures/` | Publication-facing PNG/PDF figures | Yes |
| `outputs/` | Full predictions, caches, and intermediate outputs | No |
| `sec_financial_features_*.csv` | Large derived feature table | No |
| parent project `data/` | Raw SEC quarterly files | No |

The public repository intentionally separates source-controlled scientific artifacts from large, reconstructable local data.
