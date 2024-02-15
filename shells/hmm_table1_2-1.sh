# This retrieves table 1 and 2 for claim 1 in the paper.
# Table 1 for all methods except DynaMask and ExtremalMask trained on CE is experiments/hmm/reproducibility_results/hmm_results_per_fold_averaged.csv
# Table 2 for all methods except DynaMask and ExtremalMask trained on CE is experiments/hmm/reproducibility_results/hmm_results_per_fold_ratio.csv
outputfile=hmm_results_per_fold.csv

rm -f $outputfile
python -m experiments.hmm.reset
experiments/hmm/main.sh
mv $outputfile experiments/hmm/reproducibility_results
python -m experiments.utils.compare_results --data hmm --repro-file $outputfile --ratio
