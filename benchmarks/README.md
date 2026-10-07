# Benchmarks

`results/` holds one immutable JSON per (training run, candidate model), written by
`make train-synthetic` / `make train-mimic` (`app/ml/training/benchmark.py`). Files are
never overwritten. `scripts/summarize_results.py` (`make benchmark-summary`) prints a
Markdown table for the article.

Only aggregate metrics are stored here. Never put dataset rows, identifiers or chief
complaints in this directory. Results from MIMIC-IV-ED remain subject to the PhysioNet
data use agreement: share aggregates only.

Records whose `git_commit` ends in `-dirty` were produced from uncommitted code and
should not be cited.
