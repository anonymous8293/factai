# Instructions for reproducing the results in "Reproducibility Study of 'Learning Perturbations to Explain Time Series Predictions'"

The first steps are installing conda and the associated environment.

1. **Install conda:**
   - Go to <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>.
   - Install conda.
   - Make sure that the `conda` command works in your bash script.

2. **Install environment:**
    - Either run `sh shells\install_env.sh` in your shell script or `conda install -f environment.yml`.
    - Activate the environment by `conda activate tint`.

## Reproducibility Study

The jobs file under the jobs directory under the root should be modified if one wishes to reproduce our results on a system supporting job files. Otherwise, the tables in our paper may be retrieved from running the shell scripts. Specifically, by executing the following commands in the bash script for:

- Table 1 and 2 without DynaMask and ExtremalMask trained on CE: `sh shells\hmm_table1_2-1.sh`.
  - Table 1 in `experiments/hmm/reproducibility_results/hmm_results_per_fold_averaged.csv`
  - Table 2 in `experiments/hmm/reproducibility_results/hmm_results_per_fold_ratio.csv`
- Table 1 and 2 with only DynaMask and ExtremalMask trained on CE: `sh shells\hmm_table1_2-2.sh`.
  - Table 1 in `experiments/hmm/reproducibility_results/hmm_results_per_fold_CE_averaged.csv`
  - Table 2 in `experiments/hmm/reproducibility_results/hmm_results_per_fold_CE_ratio.csv`
- Table 3 and 4: `sh shells\mimic_table3_4.sh`.
  - Table 3 is `experiments/mimic3/mortality/reproducibility_results/mimic_results_per_fold_averaged.csv`
  - Table 4 is `experiments/mimic3/mortality/reproducibility_results/mimic_results_per_fold_ratio.csv`
- Table 5 and 7: `sh shells\hmm_table5_7.sh`.
  - Table 5 is `experiments/hmm/reproducibility_results/hmm_deletion_results_per_fold_averaged.csv`
  - Table 7 is `experiments/hmm/reproducibility_results/hmm_deletion_results_per_fold_ratio.csv`
- Table 6 and 8: `sh shells\mimic_table6_8.sh`.
  - Table 3 is `experiments/mimic3/mortality/reproducibility_results/mimic_deletion_results_per_fold_averaged.csv`
  - Table 4 is `experiments/mimic3/mortality/reproducibility_results/mimic_deletion_results_per_fold_ratio.csv`

Furthermore, our results may be found under the `reproducibility_results` directory with the prefix `our`.

## Additional Study

To reproduce the tables and figures associated to the extensions, please follow the walkthroughs in the following Jupyter notebooks:
- Extension 1 and Appendix: `Extension_1.ipynb`
- Extension 2: `Extension_2.ipynb`