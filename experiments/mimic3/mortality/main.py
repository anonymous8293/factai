import multiprocessing as mp
import numpy as np
import random
import torch as th
import torch.nn as nn

from argparse import ArgumentParser
from captum.attr import DeepLift, GradientShap, IntegratedGradients, Lime
from pytorch_lightning import Trainer, seed_everything
from typing import List
from tint.utils.perturbations import compute_perturbations

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
from tint.datasets import Mimic3
from tint.metrics import (
    accuracy,
    comprehensiveness,
    cross_entropy,
    log_odds,
    sufficiency,
)
from tint.models import MLP, RNN
from experiments.utils.get_model import get_model, get_explainer, save_explainer, load_explainer2
from tint.utils.model_loading import load_explainer
from tint.utils.perturbations import compute_alternative, compute_alternative2

from experiments.mimic3.mortality.classifier import MimicClassifierNet



def output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device):
    
    
    cpu_classifier=classifier.to("cpu")
    cpu_x_test = x_test.to("cpu")

    # Dict for baselines
    baselines_dict = {0: "Average", 1: "Zeros"}
    with open(output_file, "a") as fp, lock:
        for i, baselines in enumerate([x_avg, 0.0]):
            for topk in areas:
                for k, v in attr.items():
                    acc = accuracy(
                        cpu_classifier,
                        cpu_x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    comp = comprehensiveness(
                        cpu_classifier,
                        cpu_x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    ce = cross_entropy(
                        cpu_classifier,
                        cpu_x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    l_odds = log_odds(
                        cpu_classifier,
                        cpu_x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )
                    suff = sufficiency(
                        cpu_classifier,
                        cpu_x_test,
                        attributions=v.cpu(),
                        baselines=baselines,
                        topk=topk,
                    )

                    fp.write(str(seed) + ",")
                    fp.write(str(fold) + ",")
                    fp.write(baselines_dict[i] + ",")
                    fp.write(str(topk) + ",")
                    fp.write(k + ",")
                    fp.write(str(lambda_1) + ",")
                    fp.write(str(lambda_2) + ",")
                    fp.write(f"{acc:.4},")
                    fp.write(f"{comp:.4},")
                    fp.write(f"{ce:.4},")
                    fp.write(f"{l_odds:.4},")
                    fp.write(f"{suff:.4}")
                    fp.write("\n")
    attr.clear()

    classifier=classifier.to(device)
    x_test = x_test.to(device)

def main(
    explainers: List[str],
    areas: list,
    device: str = "cpu",
    fold: int = 0,
    seed: int = 42,
    deterministic: bool = False,
    lambda_1: float = 1.0,
    lambda_2: float = 1.0,
    output_file: str = "results.csv",
    preservation_mode: bool = True
):
    dataset_name = 'mimic3'
    data_name = dataset_name
    pickle_folder="experiments/pickles/"
    
        
    retrain = False

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
    mimic3 = Mimic3(n_folds=5, fold=fold, seed=seed)



    # Create classifier
    classifier = MimicClassifierNet(
        feature_size=31,
        n_state=2,
        hidden_size=200,
        regres=True,
        loss="cross_entropy",
        lr=0.0001,
        l2=1e-3,
    )

    # Train classifier
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=device_id,
        deterministic=deterministic,
        logger=False,
    )
    classifier = get_model(trainer, classifier, 'classifier', dataset_name, seed, fold, lambda_1=lambda_1, lambda_2=lambda_2, datamodule=mimic3)

    # Get data for explainers
    with lock:
        x_train = mimic3.preprocess(split="train")["x"].to(device)
        x_test = mimic3.preprocess(split="test")["x"].to(device)
        y_test = mimic3.preprocess(split="test")["y"].to(device)

        # print("x_train shape", x_train.size())
        # print("x_test shape", x_train.size())
        # print("y_t# est shape", x_train.size())

    print("y_test mean", th.mean(y_test.float()))

    # Switch to eval
    classifier.eval()

    # Set model to device
    classifier.to(device)

    

    # Disable cudnn if using cuda accelerator.
    # Please see https://captum.ai/docs/faq#how-can-i-resolve-cudnn-rnn-backward-error-for-rnn-or-lstm-network
    # for more information.
    if accelerator == "cuda":
        th.backends.cudnn.enabled = False


    #for printing out
    # Compute x_avg for the baseline
    x_avg = x_test.mean(1, keepdim=True).repeat(1, x_test.to("cpu").shape[1], 1)
        # Classif:ier and x_test to cpu

    # Create dict of attributions
    attr = dict()
    print("explainers is ",explainers)
    if "deep_lift" in explainers:
        print("starting deeplift")
        model_name="deep_lift" 
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            print("no explainer found!!!")
            explainer = TimeForwardTunnel(DeepLift(classifier))
            attr["deep_lift"] = explainer.attribute(
                x_test,
                baselines=x_test * 0,
                task="binary",
                show_progress=True,
            ).abs()
        
        # save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        # save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)

    if "dyna_mask" in explainers:
        model_name="dyna_mask" 
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            trainer = Trainer(
                max_epochs=1000,
                accelerator=accelerator,
                devices=device_id,
                log_every_n_steps=2,
                deterministic=deterministic,
                logger=False,
            )
            mask = MaskNet(
                forward_func=classifier,
                perturbation="fade_moving_average",
                keep_ratio=list(np.arange(0.1, 0.7, 0.1)),
                deletion_mode=True,
                size_reg_factor_init=0.1,
                size_reg_factor_dilation=10000,
                time_reg_factor=0.0,
                loss="cross_entropy",
            )
            explainer = DynaMask(dataset_name, classifier.to(device), seed, fold)
            _attr = explainer.attribute(
                x_test.to(device),
                trainer=trainer,
                mask_net=mask,
                batch_size=100,
                return_best_ratio=True,
                device=device
            )
            print(f"Best keep ratio is {_attr[1]}")
            attr["dyna_mask"] = _attr[0].to(device)

            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( mask,             model_name+"_mask",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)

    if "extremal_mask" in explainers:
        model_name="extremal_mask"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            trainer = Trainer(
                max_epochs=500,
                accelerator=accelerator,
                devices=device_id,
                log_every_n_steps=2,
                deterministic=deterministic,
                logger=False,
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
                loss="cross_entropy",
                optim="adam",
                lr=0.01,
            )
            explainer = ExtremalMask(dataset_name, classifier, seed, fold)
            _attr = explainer.attribute(
                x_test,
                trainer=trainer,
                mask_net=mask,
                batch_size=100,
            )
            attr["extremal_mask"] = _attr.to(device)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)


    if "extremal_mask_alt" in explainers:
        model_name="extremal_mask_alt"
        attr[model_name]=None
        inputs_mimic=x_test
        (extremal_mask_attr_mimic,extremal_mask_explainer_mimic,extremal_mask_mask_net_mimic) = load_explainer(dataset_name="mimic3", pickle_dir="experiments/pickles/", method="extremal_mask", seed=seed, fold=fold)
        (extremal_mimic_batch,extremal_perturbation_mimic, extremal_mask_mimic, extremal_x1_mimic, extremal_x2_mimic,) = compute_perturbations(
            data=inputs_mimic,mask_net=extremal_mask_mask_net_mimic,perturb_net=extremal_mask_mask_net_mimic.net.model,batch_idx=0)
        alt_mask = compute_alternative(extremal_mimic_batch, extremal_mask_mimic, extremal_perturbation_mimic)
        attr[model_name]=alt_mask
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  


    if "extremal_mask_alt2" in explainers:
        model_name="extremal_mask_alt2"
        attr[model_name]=None
        inputs_mimic=x_test
        (extremal_mask_attr_mimic,extremal_mask_explainer_mimic,extremal_mask_mask_net_mimic) = load_explainer(dataset_name="mimic3", pickle_dir="experiments/pickles/", method="extremal_mask", seed=seed, fold=fold)
        (extremal_mimic_batch,extremal_perturbation_mimic, extremal_mask_mimic, extremal_x1_mimic, extremal_x2_mimic,) = compute_perturbations(
            data=inputs_mimic,mask_net=extremal_mask_mask_net_mimic,perturb_net=extremal_mask_mask_net_mimic.net.model,batch_idx=0)
        alt_mask = compute_alternative2(extremal_mimic_batch, extremal_mask_mimic, extremal_perturbation_mimic)
        
        attr[model_name]=alt_mask
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  




    if "fit" in explainers:
        model_name="fit"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            generator = JointFeatureGeneratorNet(rnn_hidden_size=6)
            trainer = Trainer(
                max_epochs=300,
                accelerator=accelerator,
                devices=device_id,
                log_every_n_steps=10,
                deterministic=deterministic,
                logger=False,
            )
            explainer = Fit(
                dataset_name,
                classifier,
                generator=generator,
                datamodule=mimic3,
                trainer=trainer,
                seed=seed,
                fold=fold
            )
            attr["fit"] = explainer.attribute(x_test, show_progress=True)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)

    if "gradient_shap" in explainers:
        model_name="gradient_shap"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
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
            model_name="gradient_shap"
            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  

    if "integrated_gradients" in explainers:
        model_name="integrated_gradients"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            explainer = TimeForwardTunnel(IntegratedGradients(classifier))
            attr["integrated_gradients"] = explainer.attribute(
                x_test,
                baselines=x_test * 0,
                internal_batch_size=200,
                task="binary",
                show_progress=True,
            ).abs()

            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  

    if "lime" in explainers:
        model_name="lime"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            explainer = TimeForwardTunnel(Lime(classifier))
            attr["lime"] = explainer.attribute(
                x_test,
                task="binary",
                show_progress=True,
            ).abs()

            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)

        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  

    if "augmented_occlusion" in explainers:

        model_name="augmented_occlusion"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
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
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)

    if "occlusion" in explainers:
        model_name="occlusion"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            explainer = TimeForwardTunnel(TemporalOcclusion(classifier))
            attr["occlusion"] = explainer.attribute(
                x_test,
                sliding_window_shapes=(1,),
                baselines=x_train.mean(0, keepdim=True),
                attributions_fn=abs,
                task="binary",
                show_progress=True,
            ).abs()

            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)

        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  

    if "retain" in explainers:
        model_name="retain"
        attr[model_name]=load_explainer2( dataset_name, model_name, pickle_folder, seed, fold)
        if(attr[model_name] is None):
            retain = RetainNet(
                dim_emb=128,
                dropout_emb=0.4,
                dim_alpha=8,
                dim_beta=8,
                dropout_context=0.4,
                dim_output=2,
                temporal_labels=False,
                loss="cross_entropy",
            )
            explainer = Retain(
                dataset_name,
                datamodule=mimic3,
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
            save_explainer( attr[model_name], model_name+"_attr",             dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
            save_explainer( explainer,        model_name+"_explainer",        dataset_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        output_all(output_file, x_avg,areas,  attr, classifier, x_test, lock, seed, fold, lambda_1, lambda_2, device)
        print("finished analyzing ", model_name, "output to ",output_file)  







def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--explainers",
        type=str,
        default=[
            "extremal_mask",
           "deep_lift",
           "dyna_mask",
            "augmented_occlusion",
            "occlusion",
            "retain",
            "integrated_gradients",
        ],
        nargs="+",
        metavar="N",
        help="List of explainer to use.",
    )
    parser.add_argument(
        "--areas",
        type=float,
        default=[
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ],
        nargs="+",
        metavar="N",
        help="List of areas to use.",
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
        default="mimic_results_per_fold.csv",
        help="Where to save the results.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        explainers=args.explainers,
        areas=args.areas,
        device=args.device,
        fold=args.fold,
        seed=args.seed,
        deterministic=args.deterministic,
        lambda_1=args.lambda_1,
        lambda_2=args.lambda_2,
        output_file=args.output_file,
    )
