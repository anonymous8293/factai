# This retrieves table 5 and 7 for claim 3 in the paper.
# Table 5 is experiments/hmm/reproducibility_results/hmm_deletion_results_per_fold_averaged.csv
# Table 5 is experiments/hmm/reproducibility_results/hmm_deletion_results_per_fold_ratio.csv
outputfile=hmm_deletion_results_per_fold.csv
originalfile=original_deletion_results.csv

rm -f $outputfile
python -m experiments.hmm.reset --outputfile $outputfile
experiments/hmm/main.sh --outputfile $outputfile --deletion
mv $outputfile experiments/hmm/reproducibility_results
python -m experiments.utils.compare_results --data hmm --repro-file $outputfile --original-file $originalfile --deletion --ratio
