@echo off

del hmm_results_per_fold.csv
python -m experiments.hmm.reset -e main

for /l %%x in (0, 1, 4) do (
    python -m experiments.hmm.main --explainers dyna_mask --fold %%x
)