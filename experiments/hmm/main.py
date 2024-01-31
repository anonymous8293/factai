import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from typing import List

from tint.attr import (
    DynaMask,
    ExtremalMask,
    Fit,
    Retain,
    TemporalAugmentedOcclusion,
    TemporalOcclusion,
    TimeForwardTunnel,
)
from tint.attr.models import (
    ExtremalMaskNet,
    JointFeatureGeneratorNet,
    MaskNet,
    RetainNet,
)
from tint.datasets import HMM
from tint.metrics.white_box import (
    aup,
    aur,
    information,
    entropy,
    roc_auc,
    auprc,
)
from tint.models import MLP, RNN

from experiments.utils.get_model import get_model
from experiments.hmm.classifier import StateClassifierNet
import pickle



def main(
    explainers: List[str],
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "hmm_results_per_fold.csv",
    preservation_mode: bool=True,
    preservation_test:bool=False
):
    dataset_name = 'hmm'

    # If deterministic, seed everything
    if deterministic:
        seed_everything(seed=seed, workers=True)

    # Get accelerator and device
    accelerator = device.split(":")[0]
    device_id = 1
    if len(device.split(":")) > 1:
        device_id = [int(device.split(":")[1])]

    # Create lock
    lock = mp.Lock()

    # Load data
    hmm = HMM(n_folds=5, fold=fold, seed=seed)

    # Create classifier
    classifier = StateClassifierNet(
        feature_size=3,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=False
    )
    
    classifier = get_model(trainer, classifier, 'classifier', dataset_name, seed, fold, lambda_1=lambda_1, lambda_2=lambda_2, datamodule=hmm)
    
    # trainer.fit(classifier, datamodule=hmm)

    # Get data for explainers
    with lock:
        x_train = hmm.preprocess(split="train")["x"].to(device)
        x_test = hmm.preprocess(split="test")["x"].to(device)
        y_test = hmm.preprocess(split="test")["y"].to(device)
        true_saliency = hmm.true_saliency(split="test").to(device)

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-nn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False

    # Create dict of attributions
    attr = dict()

    if "deep_lift" in explainers:
        explainer = TimeForwardTunnel(DeepLift(classifier))
        attr["deep_lift"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            task="binary",
            show_progress=True,
        ).abs()

        with open(output_file, "a") as fp, lock:
            k = "deep_lift"
            v = attr["deep_lift"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
        
    if "dyna_mask" in explainers:
        trainer = Trainer(
            max_epochs=1000,
            accelerator=accelerator,
            devices=device_id,
            deterministic=deterministic,
            logger=False
        )
        mask = MaskNet(
            forward_func=classifier,
            perturbation="gaussian_blur",
            sigma_max=1,
            keep_ratio=list(np.arange(0.25, 0.35, 0.01)),
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            time_reg_factor=1.0,
        )
        explainer = DynaMask(dataset_name, classifier, seed, fold)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
            return_best_ratio=True,
        )
        print(f"Best keep ratio is {_attr[1]}")
        attr["dyna_mask"] = _attr[0].to(device)

        with open(output_file, "a") as fp, lock:
            k = "dyna_mask"
            v = attr["dyna_mask"]
            fp.write(str(seed) + ",")
            if(preservation_test):
                fp.write(str(preservation_mode)+ ",")
                fp.write(str(not preservation_mode)+ ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")

    if "extremal_mask" in explainers:
        trainer = Trainer(
            max_epochs=500,
            accelerator=accelerator,
            devices=device_id,
            deterministic=deterministic,
            logger=False
        )
        mask = ExtremalMaskNet(
            forward_func=classifier,
            preservation_mode=preservation_mode,
            model=nn.Sequential(
                RNN(
                    input_size=x_test.shape[-1],
                    rnn="gru",
                    hidden_size=x_test.shape[-1],
                    bidirectional=True,
                ),
                MLP([2 * x_test.shape[-1], x_test.shape[-1]]),
            ),
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            optim="adam",
            lr=0.01,
        )
        explainer = ExtremalMask(dataset_name, classifier, seed, fold)
        _attr = explainer.attribute(
            x_test,
            additional_forward_args=(True,),
            trainer=trainer,
            mask_net=mask,
            batch_size=100,
        )
        attr["extremal_mask"] = _attr.to(device)

        with open(output_file, "a") as fp, lock:
            k = "extremal_mask"
            v = attr["extremal_mask"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")

    if "fit" in explainers:
        generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
        trainer = Trainer(
            max_epochs=300,
            accelerator=accelerator,
            devices=device_id,
            deterministic=deterministic,
            logger=False
        )
        explainer = Fit(
            dataset_name,
            classifier,
            generator=generator,
            datamodule=hmm,
            trainer=trainer,
            seed=seed,
            fold=fold
        )
        attr["fit"] = explainer.attribute(x_test, show_progress=True)

        with open(output_file, "a") as fp, lock:
            k = "fit"
            v = attr["fit"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")

    if "gradient_shap" in explainers:
        explainer = TimeForwardTunnel(GradientShap(classifier.cpu()))
        attr["gradient_shap"] = explainer.attribute(
            x_test.cpu(),
            baselines=th.cat([x_test.cpu() * 0, x_test.cpu()]),
            n_samples=50,
            stdevs=0.0001,
            task="binary",
            show_progress=True,
        ).abs().to(device)
        classifier.to(device)

        with open(output_file, "a") as fp, lock:
            k = "gradient_shap"
            v = attr["gradient_shap"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")

    if "integrated_gradients" in explainers:
        explainer = TimeForwardTunnel(IntegratedGradients(classifier))
        attr["integrated_gradients"] = explainer.attribute(
            x_test,
            baselines=x_test * 0,
            internal_batch_size=200,
            task="binary",
            show_progress=True,
        ).abs()

        with open(output_file, "a") as fp, lock:
            k = "integrated_gradients"
            v = attr["integrated_gradients"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
        
    if "lime" in explainers:
        explainer = TimeForwardTunnel(Lime(classifier))
        attr["lime"] = explainer.attribute(
            x_test,
            task="binary",
            show_progress=True,
        ).abs()

        with open(output_file, "a") as fp, lock:
            k = "lime"
            v = attr["lime"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
    
    if "augmented_occlusion" in explainers:
        explainer = TimeForwardTunnel(
            TemporalAugmentedOcclusion(
                classifier, data=x_train, n_sampling=10, is_temporal=True
            )
        )
        attr["augmented_occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

        with open(output_file, "a") as fp, lock:
            k = "augmented_occlusion"
            v = attr["augmented_occlusion"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
        
    if "occlusion" in explainers:
        explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
        attr["occlusion"] = explainer.attribute(
            x_test,
            sliding_window_shapes=(1,),
            baselines=x_train.mean(0, keepdim=True),
            attributions_fn=abs,
            task="binary",
            show_progress=True,
        ).abs()

        with open(output_file, "a") as fp, lock:
            k = "occlusion"
            v = attr["occlusion"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
        
    if "retain" in explainers:
        retain = RetainNet(
            dim_emb=128,
            dropout_emb=0.4,
            dim_alpha=8,
            dim_beta=8,
            dropout_context=0.4,
            dim_output=2,
            loss="cross_entropy",
        )
        explainer = Retain(
            dataset_name,
            datamodule=hmm,
            retain=retain,
            trainer=Trainer(
                max_epochs=50,
                accelerator=accelerator,
                devices=device_id,
                deterministic=deterministic,
                logger=False
            ),
            seed=seed,
            fold=fold
        )
        attr["retain"] = (
            explainer.attribute(x_test, target=y_test).abs().to(device)
        )

        with open(output_file, "a") as fp, lock:
            k = "retain"
            v = attr["retain"]
            fp.write(str(seed) + ",")
            fp.write(str(fold) + ",")
            fp.write(k + ",")
            fp.write(str(lambda_1) + ",")
            fp.write(str(lambda_2) + ",")
            fp.write(f"{aup(v, true_saliency):.4},")
            fp.write(f"{aur(v, true_saliency):.4},")
            fp.write(f"{information(v, true_saliency):.4},")
            fp.write(f"{entropy(v, true_saliency):.4},")
            fp.write(f"{roc_auc(v, true_saliency):.4},")
            fp.write(f"{auprc(v, true_saliency):.4}")
            fp.write("\n")
        
    # with open(output_file, "a") as fp, lock:
    #     for k, v in attr.items():
    #         fp.write(str(seed) + ",")
    #         fp.write(str(fold) + ",")
    #         fp.write(k + ",")
    #         fp.write(str(lambda_1) + ",")
    #         fp.write(str(lambda_2) + ",")
    #         fp.write(f"{aup(v, true_saliency):.4},")
    #         fp.write(f"{aur(v, true_saliency):.4},")
    #         fp.write(f"{information(v, true_saliency):.4},")
    #         fp.write(f"{entropy(v, true_saliency):.4},")
    #         fp.write(f"{roc_auc(v, true_saliency):.4},")
    #         fp.write(f"{auprc(v, true_saliency):.4}")
    #         fp.write("\n")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            #"deep_lift",
            "dyna_mask",
            "extremal_mask",
            "fit",
            "gradient_shap",
            "integrated_gradients",
            "lime",
            "augmented_occlusion",
            "occlusion",
            "retain",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Which device to use.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold of the cross-validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for data generation.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Whether to make training deterministic or not.",
    )
    parser.add_argument(
        "--lambda-1",
        type=float,
        default=1.0,
        help="Lambda 1 hyperparameter.",
    )
    parser.add_argument(
        "--lambda-2",
        type=float,
        default=1.0,
        help="Lambda 2 hyperparameter.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="hmm_results_per_fold.csv",
        help="Where to save the results.",
    )
    parser.add_argument(
        "--deletion-mode",
        action="store_false",
        help="By default uses extremal_mask preservation game. When specified, it runs for the deletion game.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        output_file=args.output_file,
        preservation_mode=args.deletion_mode
    )
