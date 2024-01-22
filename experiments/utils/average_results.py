import os
import pandas as pd
from argparse import ArgumentParser

def average_main_experiment(results_file='results_per_fold.csv'):
    results_file_wo_extension = os.path.splitext(results_file)[0]

    # Read the CSV file
    df = pd.read_csv(results_file)

    # Remove the last two columns
    df = df.iloc[:, :-2]

    # Group by 'Seed', 'Explainer', 'Lambda_1', and 'Lambda_2'
    grouped_df = df.groupby(['Seed', 'Explainer', 'Lambda_1', 'Lambda_2']).agg({
        'AUP': ['mean', 'std'],
        'AUR': ['mean', 'std'],
        'Information': ['mean', 'std'],
        'Entropy': ['mean', 'std']
    }).reset_index()
    
    # Flatten the MultiIndex column headers
    grouped_df.columns = ['_'.join(col).strip() for col in grouped_df.columns.values]

    # Create columns with average ± std format
    average_df = pd.DataFrame()
    average_df[['Seed', 'Explainer', 'Lambda_1', 'Lambda_2']] = grouped_df[['Seed_', 'Explainer_', 'Lambda_1_', 'Lambda_2_']]

    for metric in ['AUP', 'AUR', 'Information', 'Entropy']:
        average_df[f'{metric}'] = grouped_df.apply(lambda row: f"{row[f'{metric}_mean']:.2f} ± {row[f'{metric}_std']:.4f}", axis=1)

    # Save the result to a new CSV file with column headers
    average_df.to_csv(f'{results_file_wo_extension}_averaged.csv', index=False)
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        default='main',
        help="Experiment to obtain average result for.",
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
    if args.experiment == 'main':
        average_main_experiment(args.results_file)
    
