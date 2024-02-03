import os
import torch as th


import pickle
# def save_explainer(explainer, explainer_name="explainer"):
#     with open(f"{explainer_name}.pkl", "wb") as f:
#         pickle.dump(explainer, f)

def load_explainer2(
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
            print(f"--------- Could not find and load {path}")
    return load_pickle_file(f"{pickle_dir}{dataset_name}_{method}_attr_{seed}_{fold}")


def get_explainer_checkpoint(model_name, data_name, seed, fold, lambda_1 = None, lambda_2 = None, retrain = False, preservation_mode = True):
    checkpoint_dir = 'experiments/pickles'
    if model_name != 'extremal_mask':
        checkpoint_path = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{fold}.ckpt'
    elif model_name == 'extremal_mask' and (lambda_1 is None or lambda_2 is None):
        raise ValueError('must specify lambda 1 and lambda 2 values for extremal mask explainer')
    else:
        if preservation_mode:
            checkpoint_path = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}.ckpt'
        else:
            checkpoint_path = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}_deletion.ckpt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_path

def get_explainer(model_name, data_name, seed, fold, lambda_1 = None, lambda_2 = None, retrain = False, preservation_mode = True):
    checkpoint_path=get_explainer_checkpoint( model_name, data_name, seed, fold, lambda_1 , lambda_2 , retrain , preservation_mode)
    if os.path.exists(checkpoint_path) and not retrain:
        # model.load_state_dict(th.load(checkpoint_path))
        print(f'{model_name} will load the explainer with fold = {fold}, if applicable, lambda 1 = {lambda_1} and lambda 2 = {lambda_2}.')
        with open(f"{checkpoint_path}", "rb") as f:
            attr = pickle.load(f)
        return attr
    else:
        return None

def save_explainer( explainer, model_name, data_name, seed, fold, lambda_1 = None, lambda_2 = None, retrain = False, preservation_mode = True):
    checkpoint_path=get_explainer_checkpoint( model_name, data_name, seed, fold, lambda_1 , lambda_2 , retrain , preservation_mode)
    if retrain or not os.path.exists(checkpoint_path) :
        with open(f"{checkpoint_path}", "wb") as f:
            pickle.dump(explainer, f)


def get_model(trainer, model, model_name, data_name, seed, fold, train_dataloaders = None, datamodule = None, lambda_1 = None, lambda_2 = None, retrain = False, preservation_mode = True, save_as_pth=False):

    checkpoint_dir = 'experiments/checkpoints'

    if model_name != 'extremal_mask':
        checkpoint_path_without_extension = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{fold}'
    elif model_name == 'extremal_mask' and (lambda_1 is None or lambda_2 is None):
        raise ValueError('must specify lambda 1 and lambda 2 values for extremal mask explainer')
    else:
        if preservation_mode:
            checkpoint_path_without_extension = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}'
        else:
            checkpoint_path_without_extension = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}_deletion'

    checkpoint_path = checkpoint_path_without_extension + '.ckpt'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.exists(checkpoint_path) and not retrain:
        checkpoint = th.load(checkpoint_path)
        model.load_state_dict(checkpoint)
        print(f'{model_name} has loaded the saved weights on {data_name} dataset with fold = {fold}, if applicable, lambda 1 = {lambda_1} and lambda 2 = {lambda_2}.')
        if model_name != 'classifier':
            full_model_name = model_name+"_mask"
        else:
            full_model_name = model_name
        checkpoint_path = checkpoint_path_without_extension + '.pkl'
        save_explainer( model,             full_model_name,             data_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
        if hasattr(model, 'net')  and hasattr(model.net, 'model'):
            save_explainer( model.net.model,   model_name+"_perturbation_net", data_name, seed, fold, lambda_1, lambda_2, retrain, preservation_mode)
    else:
        # Fit and retain training
        if datamodule is not None and train_dataloaders is not None:
            trainer.fit(model, train_dataloaders=train_dataloaders, datamodule=datamodule)
        # GRU classifier training
        elif datamodule is not None:
            trainer.fit(model, datamodule=datamodule)
        # Dyna and extremal mask training
        elif train_dataloaders is not None:
            trainer.fit(model, train_dataloaders=train_dataloaders)

        th.save(model.state_dict(), checkpoint_path)

    return model