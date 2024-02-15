import os
import pickle

import torch as th

from experiments.hmm.classifier import StateClassifier, StateClassifierNet
from experiments.mimic3.mortality.classifier import MimicClassifierNet


def get_classifier(
    experiment: str,
    seed: int or str,
    fold: int or str,
    checkpoint_dir: str = "experiments/checkpoints/",
):
    classifier = None
    if experiment == "hmm":
        # From HMM experiments
        classifier = StateClassifierNet(
            feature_size=3,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
        )
    elif experiment == "mimic3":
        # From MIMIC-III mortality experiment
        classifier = MimicClassifierNet(
            feature_size=31,
            n_state=2,
            hidden_size=200,
            regres=True,
            loss="cross_entropy",
            lr=0.0001,
            l2=1e-3,
        )

    path = f"{checkpoint_dir}{experiment}_classifier_{seed}_{fold}.ckpt"
    classifier.load_state_dict(th.load(path))
    return classifier


def save_explainer(explainer, explainer_name="explainer"):
    with open(f"{explainer_name}.pkl", "wb") as f:
        pickle.dump(explainer, f)


def load_explainer(
    dataset_name: str,
    method: str,
    pickle_dir: str = "experiments/checkpoints/",
    seed: int = 42,
    fold: int = 0,
):
    def load_pickle_file(path: str):
        try:
            return th.load(f"{path}.pt")
        except FileNotFoundError:
            pass

        try:
            return th.load(f"{path}.pkl")
        except FileNotFoundError:
            pass

        try:
            return th.load(f"{path}.ckpt")
        except FileNotFoundError:
            print(f"Could not find and load {path}")

    attr = load_pickle_file(f"{pickle_dir}{dataset_name}_{method}_attr_{seed}_{fold}")
    explainer = load_pickle_file(
        f"{pickle_dir}{dataset_name}_{method}_explainer_{seed}_{fold}"
    )
    mask_net = load_pickle_file(
        f"{pickle_dir}{dataset_name}_{method}_mask_{seed}_{fold}"
    )

    return attr.detach(), explainer, mask_net


def get_model(
    trainer,
    model,
    model_name,
    data_name,
    seed,
    fold,
    train_dataloaders=None,
    datamodule=None,
    lambda_1=None,
    lambda_2=None,
    retrain=False,
    preservation_mode=True,
    checkpoint_dir: str = "experiments/checkpoints",
):

    if model_name != "extremal_mask":
        checkpoint_path = (
            f"{checkpoint_dir}/{data_name}_{model_name}_{seed}_{fold}.ckpt"
        )
    elif model_name == "extremal_mask" and (lambda_1 is None or lambda_2 is None):
        raise ValueError(
            "must specify lambda 1 and lambda 2 values for extremal mask explainer"
        )
    else:
        if preservation_mode:
            checkpoint_path = f"{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}.ckpt"
        else:
            checkpoint_path = f"{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}_deletion.ckpt"

    print(f"Trying to load {checkpoint_path}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.exists(checkpoint_path) and not retrain:
        model.load_state_dict(th.load(checkpoint_path))
        print(
            f"{model_name} has loaded the saved weights on {data_name} dataset with fold = {fold}, if applicable, lambda 1 = {lambda_1} and lambda 2 = {lambda_2}."
        )
    else:
        # Fit and retain training
        if datamodule is not None and train_dataloaders is not None:
            trainer.fit(
                model, train_dataloaders=train_dataloaders, datamodule=datamodule
            )
        # GRU classifier training
        elif datamodule is not None:
            trainer.fit(model, datamodule=datamodule)
        # Dyna and extremal mask training
        elif train_dataloaders is not None:
            trainer.fit(model, train_dataloaders=train_dataloaders)
        else:
            raise Exception(
                f"You are trying to retrain {model_name} but datamodule or train_dataloaders does not exist"
            )

        th.save(model.state_dict(), checkpoint_path)

    return model
