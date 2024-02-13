import os
import pandas as pd
from argparse import ArgumentParser

def format_with_condition(value, std, precision=2):
    if -1 < value < 1:
        return f"{value:.{precision+1}f}±{std:.{precision+1}f}"
    elif -10 < value < 10:
        return f"{value:.{precision}f}±{std:.{precision}f}"
    elif -100 < value < 100:
        return f"{value:.{precision-1}f}±{std:.{precision-1}f}"
    elif -1000 < value < 1000:
        return f"{value:.{precision-2}f}±{std:.{precision-2}f}"
    else:
        return f"{value:.{precision}e}±{std:.{precision}e}"


def average_main_experiment(data, results_path='results_per_fold.csv', deletion_original=False):
    if data == 'hmm':
        metrics_columns = ['AUP', 'AUR', 'Information', 'Entropy']
        grouping_columns = ['Seed', 'Explainer', 'Lambda_1', 'Lambda_2']
    elif data == 'mimic':
        metrics_columns = ['Accuracy', 'Comprehensiveness', 'Cross Entropy', 'Sufficiency']
        grouping_columns = ['Seed', 'Explainer', 'Lambda_1', 'Lambda_2', 'Topk', 'Baseline']

    results_path_wo_extension = os.path.splitext(results_path)[0]

    # Read the CSV file
    df = pd.read_csv(results_path)

    if 'Topk' in df and 'Baseline' in df:
        df = df[(df['Topk'] == 0.2) & (df['Baseline'] == 'Average')]
    
    if deletion_original:
        df = df[df['Deletion'] == True]

    non_grouping_columns = [col for col in df.columns if col not in grouping_columns and col != 'Fold']

    agg_dict = {col: ['mean', 'std'] for col in non_grouping_columns}

    # Grouping and aggregating for non-grouping columns
    grouped_df = df.groupby(grouping_columns).agg(agg_dict).reset_index()
    
    # Flatten the MultiIndex column headers
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

    # Create columns with average ± std format with the conditional formatting
    average_df = pd.DataFrame()
    average_df[grouping_columns] = grouped_df[[f'{col}_' for col in grouping_columns]]

    for metric in metrics_columns:
        average_df[f'{metric}'] = grouped_df.apply(lambda row: format_with_condition(row[f'{metric}_mean'], row[f'{metric}_std']), axis=1)

    # Save the result to a new CSV file with column headers
    average_df.to_csv(f'{results_path_wo_extension}_averaged.csv', index=False)

    return average_df
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default='hmm',
        help="Experiment to obtain the average result for.",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="results_per_fold.csv",
        help="Where the results per fold are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    average_main_experiment(args.data, args.results_file)

# python -m experiments.utils.average_results --data mimic --results-file experiments\mimic3\mortality\reproducibility_results\results.csv
# python -m experiments.utils.average_results --data hmm