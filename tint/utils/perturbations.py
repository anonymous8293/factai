import torch as th
from torch.nn.functional import normalize


def compute_perturbations(
    data, mask_net, perturb_net=None, batch_idx: int or None = 0
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """Use the trained mask network to compute perturbations and their intermediate results.

    This function only works for explainer methods that train a mask network. Such methods include DynaMask and ExtremalMask.

    :param data: all data to be investigated. A subset of it will be investigated depending on the batch_idx and batch_size of the mask_net
    :param mask_net: the trained mask network
    :param perturb_net: the trained perturbation network. You can usually access it via mask_net.net.model. Defaults to None in which case the baseline will be zeroes
    :param batch_idx: the batch number to investigate, batch size will be inferred from mask_net. If set to None, all supplied data will be considered. Defaults to 0
    :return: a 5-tuple:
    - tensor of the batch of data that was used for calculation
    - tensor of perturbations predicted by perturb_net (or defaulted to)
    - tensor of the predicted mask by mask_net
    - tensor of inputs with important features masked, i.e. from the paper: data * mask + (1 - mask) * NN(data)
    - tensor of inputs with unimportant features masked, i.e.: data * (1 - mask) + NN(data) * data
    """
    if batch_idx == None:
        FROM_B = 0
        TO_B = len(data)
    else:
        batch_size = mask_net.net.batch_size
        FROM_B = batch_size * batch_idx
        TO_B = batch_size * (batch_idx + 1)

    batch = data[FROM_B:TO_B]

    # If there is a model, NN(x), that predicts perturbations, use it
    # Otherwise, default to zeroes
    baseline = perturb_net(batch) if perturb_net is not None else th.zeros_like(batch)

    # Retrieve the learned mask for the batch in question
    mask = mask_net.net.mask
    mask = mask[FROM_B:TO_B]
    mask = mask.clamp(0, 1)

    # Mask data according to samples
    # We eventually cut samples up to x time dimension
    # x1 represents inputs with important features masked.
    # x2 represents inputs with unimportant features masked.
    mask = mask[:, : batch.shape[1], ...]
    x1 = batch * mask + baseline * (1.0 - mask)
    x2 = batch * (1.0 - mask) + baseline * mask

    return batch, baseline.detach(), mask.detach(), x1.detach(), x2.detach()


def compute_alternative(batch, mask, perturbation):
    """Compute alternative saliency metric.

    1 - (1 - m) * |NN(x) - x|
    Intuition: the higher this value, the more perturbed the data is.
    """
    normalized_alt = th.abs(perturbation - batch)
    normalized_alt = normalize(normalized_alt, dim=1)
    alternative = 1 - (1 - mask) * normalized_alt
    return alternative




def compute_alternative2(batch, mask, perturbation):
    """Compute alternative saliency metric.

    1 - (1 - m) * |NN(x) - x|
    Intuition: the higher this value, the more perturbed the data is.
    """

    new_val = mask*batch +(1-mask)*perturbation
    new_mask = new_val / batch
    return mask


    # normalized_alt = th.abs(perturbation - batch)
    # normalized_alt = normalize(normalized_alt, dim=1)
    # alternative = 1 - (1 - mask) * normalized_alt
    # return alternative