import os
import torch as th

def get_model(trainer, model, model_name, data_name, seed, fold, train_dataloaders = None, datamodule = None, lambda_1 = None, lambda_2 = None, retrain = False):

    checkpoint_dir = 'experiments/checkpoints'

    if model_name != 'extremal_mask':
        checkpoint_path = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{fold}.ckpt'
    elif model_name == 'extremal_mask' and (lambda_1 is None or lambda_2 is None):
        raise ValueError('must specify lambda 1 and lambda 2 values for extremal mask explainer')
    else:
        checkpoint_path = f'{checkpoint_dir}/{data_name}_{model_name}_{seed}_{lambda_1}_{lambda_2}_{fold}.ckpt'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if os.path.exists(checkpoint_path) and not retrain:
        model.load_state_dict(th.load(checkpoint_path))
        print(f'{model_name} has loaded the saved weights on {data_name} dataset with, if applicable, lambda 1 = {lambda_1} and lambda 2 = {lambda_2}.')
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