import pandas as pd
from argparse import ArgumentParser
import numpy as np
from experiments.utils.average_results import average_results
import os

def format_with_condition(value, precision=2):
    if -1 < value < 1:
        return f"{value:.{precision+1}f}"
    elif -10 < value < 10:
        return f"{value:.{precision}f}"
    elif -100 < value < 100:
        return f"{value:.{precision-1}f}"
    elif -1000 < value < 1000:
        return f"{value:.{precision-2}f}"
    else:
        return f"{value:.{precision}e}"

def get_difference(x, y):
    mean_std_x = x.split('±')
    mean_x = mean_std_x[0]
    std_x = mean_std_x[1]
    diff = format_with_condition(abs(float(mean_x) - float(y.split('±')[0])))
    return f'{diff} ({std_x})'

def get_ratio(x, y):
    mean_std_x = str(x).split('±')
    mean_x = mean_std_x[0]
    std_x = float(mean_std_x[1])
    diff = abs(float(mean_x) - float(y.split('±')[0]))
    ratio = format_with_condition(diff/std_x)
    return ratio

def compare_results(data, original_results_filename, repro_results_filename, ratio_mode: bool = False, deletion: bool = False):
    if data == 'hmm':
        original_results_dir_path = 'experiments/hmm'
        columns_to_compare = ['AUP', 'AUR', 'Information', 'Entropy']
    elif data == 'mimic':
        columns_to_compare = ['Accuracy', 'Comprehensiveness', 'Cross Entropy', 'Sufficiency']
        original_results_dir_path = 'experiments/mimic3/mortality'

    repro_filename_without_extension = repro_results_filename.split('.')[0]
    original_results_path = f'{original_results_dir_path}/{original_results_filename}'
    output_dir_path = f'{original_results_dir_path}/reproducibility_results'
    repro_file_path = f'{output_dir_path}/{repro_results_filename}'

    if ratio_mode:
        vec_func = np.vectorize(get_ratio)
        comparison_type = 'ratio'
    else:   
        vec_func = np.vectorize(get_difference)
        comparison_type = 'diff'

    # Need to copy paste updated result from the e-mail
    if data != 'hmm':
        original_df_averaged = average_results(data, original_results_path, deletion)
    else:
        original_results_path_wo_extension = os.path.splitext(original_results_path)[0]
        original_df_averaged = pd.read_csv(f'{original_results_path_wo_extension}_averaged.csv', dtype={'Entropy': 'object'})

    repro_df_averaged = average_results(data, repro_file_path)

    explainer_values_repro = repro_df_averaged["Explainer"].unique()

    original_df_averaged_to_comp = original_df_averaged[
        original_df_averaged["Explainer"].isin(explainer_values_repro)
    ][columns_to_compare]
    repro_df_averaged_to_comp = repro_df_averaged[columns_to_compare]

    output_df = pd.DataFrame(vec_func(original_df_averaged_to_comp, repro_df_averaged_to_comp), columns=[columns_to_compare])
    output_df.insert(0, 'Explainer', repro_df_averaged['Explainer'].reset_index(drop=True))

    # Write the output dataframe to a new CSV file
    output_df.to_csv(f'{output_dir_path}/{repro_filename_without_extension}_{comparison_type}.csv', index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="hmm",
        help="The dataset used will locate the folder under experiments.",
    )
    parser.add_argument(
        "--original-file",
        type=str,
        default="original_results.csv",
        help="File of original results.",
    )
    parser.add_argument(
        "--repro-file",
        type=str,
        default="results_per_fold.csv",
        help="File of reproduced results.",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Output ratio of difference to std.",
    )
    parser.add_argument(
        "--deletion",
        action="store_true",
        help="Deletion game results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_results(args.data, args.original_file, args.repro_file, args.ratio, args.deletion)

# python -m experiments.utils.compare_results --data mimic --ratio
# python -m experiments.utils.compare_results --data hmm --original-results original_deletion_results.csv --repro_file deletion_game.csv --ratio --deletion
# python -m experiments.utils.compare_results --data hmm --ratio