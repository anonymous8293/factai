from argparse import ArgumentParser


def main(outputfile: str = 'mimic_results_per_fold.csv'):
    with open(outputfile, "w") as fp:
        fp.write(
            "Seed,Fold,Baseline,Topk,Explainer,Lambda_1,Lambda_2,Accuracy,Comprehensiveness,Cross Entropy,"
            "Log Odds,Sufficiency\n"
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--outputfile",
        type=str,
        default='mimic_results_per_fold.csv',
        help="Output file name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(outputfile=args.outputfile)
