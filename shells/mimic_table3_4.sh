# This retrieves table 3 and 4 for claim 2 in the paper.
# Table 3 is experiments/mimic3/mortality/reproducibility_results/mimic_results_per_fold_averaged.csv
# Table 4 is experiments/mimic3/mortality/reproducibility_results/mimic_results_per_fold_ratio.csv
outputfile=mimic_results_per_fold.csv

rm -f $outputfile
python -m experiments.mimic3.mortality.reset
experiments/mimic3/mortality/main.sh
mv $outputfile experiments/mimic3/mortality/reproducibility_results
python -m experiments.utils.compare_results --data mimic --repro-file $outputfile --ratio
