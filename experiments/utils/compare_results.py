import pandas as pd
from argparse import ArgumentParser
import numpy as np

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
    print("x is ",x)
    mean_std_x = x.split('±')
    mean_x = mean_std_x[0]
    std_x = mean_std_x[1]
    diff = format_with_condition(abs(float(mean_x) - float(y.split('±')[0])))
    return f'{diff} ({std_x})'

def get_ratio(x, y):
    mean_std_x = x.split('±')
    mean_x = mean_std_x[0]
    std_x = float(mean_std_x[1])
    diff = abs(float(mean_x) - float(y.split('±')[0]))
    ratio = format_with_condition(diff/std_x)
    return ratio

def compare_results(dataset_name, original_results_filename, repro_results_filename, ratio_mode: bool = False):
    repro_filename_without_extension = repro_results_filename.split('.')[0]
    output_dir_path='.'
    original_results_dir_path='.'

    repro_results_path = f'{repro_results_filename}'
    repro_results_path = f'{output_dir_path}/{repro_results_filename}'
    original_results_path = f'{original_results_dir_path}/{original_results_filename}'

    repro_df = pd.read_csv(repro_results_path)
    original_df = pd.read_csv(original_results_path)
    
    exclude_columns = ['Seed', 'Explainer', 'Lambda_1', 'Lambda_2']
    columns_to_compare = [col for col in original_df.columns if col not in exclude_columns]

    if ratio_mode:
        vec_func = np.vectorize(get_ratio)
        comparison_type = 'ratio'
    else:   
        vec_func = np.vectorize(get_difference)
        comparison_type = 'diff'

    output_df = pd.DataFrame(vec_func(original_df[columns_to_compare], repro_df[columns_to_compare]), columns=[columns_to_compare])
    output_df.insert(0, 'Explainer', repro_df['Explainer'])

    # Write the output dataframe to a new CSV file
    output_df.to_csv(f'{output_dir_path}/{repro_filename_without_extension}_{comparison_type}.csv', index=False)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hmm",
        help="The dataset used will locate the folder under experiments.",
    )
    parser.add_argument(
        "--original-results",
        type=str,
        default="original_results_averaged.csv",
        help="File of original results.",
    )
    parser.add_argument(
        "--repro_results",
        type=str,
        default="results_per_fold_averaged.csv",
        help="File of reproduced results.",
    )
    parser.add_argument(
        "--ratio",
        action="store_true",
        help="Output ratio of difference to std.",
    )

    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_results(args.dataset, args.original_results, args.repro_results, args.ratio)
