import torch as th
import torch.nn as nn
from tint.attr.models import ExtremalMaskNet, MaskNet
from tint.models import Net
from torch.nn.functional import normalize


def compute_perturbations(
    data: th.Tensor,
    mask_net: Net,
    perturb_net: nn.Module = None,
    batch_idx: int or None = 0,
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
    if isinstance(mask_net, MaskNet):
        return _compute_perturbations_dynamask(data, mask_net, batch_idx)
    elif isinstance(mask_net, ExtremalMaskNet):
        return _compute_perturbations_extremal(data, mask_net, perturb_net, batch_idx)
    else:
        raise NotImplementedError(
            "Perturbations can only be computed for ExtremalMask and DynaMask methods but mask_net is neither"
        )


def _compute_perturbations_extremal(
    data: th.Tensor,
    mask_net: ExtremalMaskNet,
    perturb_net: nn.Module = None,
    batch_idx: int or None = 0,
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    if batch_idx is None:
        from_b = 0
        to_b = len(data)
    else:
        batch_size = mask_net.net.batch_size
        from_b = batch_size * batch_idx
        to_b = batch_size * (batch_idx + 1)

    batch = data[from_b:to_b]

    # If there is a model, NN(x), that predicts perturbations, use it
    # Otherwise, default to zeroes
    baseline = perturb_net(batch) if perturb_net is not None else th.zeros_like(batch)

    # Retrieve the learned mask for the batch in question
    mask = mask_net.net.mask
    mask = mask[from_b:to_b]
    mask = mask.clamp(0, 1)

    # Mask data according to samples
    # We eventually cut samples up to x time dimension
    # x1 represents inputs with important features masked.
    # x2 represents inputs with unimportant features masked.
    mask = mask[:, : batch.shape[1], ...]
    x1 = batch * mask + baseline * (1.0 - mask)
    x2 = batch * (1.0 - mask) + baseline * mask

    return batch, baseline.detach(), mask.detach(), x1.detach(), x2.detach()


def _compute_perturbations_dynamask(
    data: th.Tensor, mask_net: MaskNet, batch_idx: int
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    if mask_net.net.perturbation == "fade_moving_average":
        return _fade_moving_average(x=data, mask_net=mask_net, batch_idx=batch_idx)
    else:
        raise NotImplementedError(
            "Only fade_moving_average is implemented for fixed-function perturbation"
        )


def _fade_moving_average(
    x: th.Tensor, mask_net: MaskNet, batch_idx: int
) -> (th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor):
    """Computes perturbation components using fade moving average.

    Adapted from MaskNet class.
    """
    net = mask_net.net
    mask = 1.0 - net.mask if net.deletion_mode else net.mask

    # Do not use net.keep_ratio here because we only use the first
    # ratio
    from_b = net.batch_size * batch_idx
    to_b = net.batch_size * (batch_idx + 1)
    mask = mask[from_b:to_b]
    x = x[from_b:to_b]

    moving_average = th.mean(x, 1).unsqueeze(1)
    x1 = mask * x + (1 - mask) * moving_average
    x2 = x * (1 - mask) + moving_average * mask
    return x.detach(), moving_average.detach(), mask.detach(), x1.detach(), x2.detach()


def compute_alternative(
    batch: th.Tensor, mask: th.Tensor, perturbation: th.Tensor
) -> th.Tensor:
    """Compute alternative saliency metric.

    1 - (1 - m) * |NN(x) - x|
    Intuition: the higher this value, the more perturbed the data is.
    """
    normalized_alt = th.abs(perturbation - batch)
    normalized_alt = normalize(normalized_alt, dim=1)
    alternative = 1 - (1 - mask) * normalized_alt
    return alternative
