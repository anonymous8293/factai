# This retrieves table 6 and 8 for claim 3 in the paper.
# Table 3 is experiments/mimic3/mortality/reproducibility_results/mimic_deletion_results_per_fold_averaged.csv
# Table 4 is experiments/mimic3/mortality/reproducibility_results/mimic_deletion_results_per_fold_ratio.csv
outputfile=mimic_deletion_results_per_fold.csv
originalresults=original_deletion_results.csv

rm -f $outputfile
python -m experiments.mimic3.mortality.reset --outputfile $outputfile
experiments/mimic3/mortality/main.sh --outputfile $outputfile --deletion
mv $outputfile experiments/mimic3/mortality/reproducibility_results
python -m experiments.utils.compare_results --data mimic --repro-file $outputfile --original-file $originalfile --deletion --ratio
