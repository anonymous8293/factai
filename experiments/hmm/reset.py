from argparse import ArgumentParser


def main(outputfile: str = 'hmm_results_per_fold.csv'):
    with open(outputfile, "w") as fp:
        fp.write(
            "Seed,Fold,Explainer,Lambda_1,Lambda_2,AUP,AUR,Information,Entropy,AUROC,AUPRC\n"
        )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--outputfile",
        type=str,
        default='hmm_results_per_fold.csv',
        help="Output file name",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(outputfile=args.outputfile)