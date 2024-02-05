import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th
from matplotlib.lines import Line2D
from scipy.stats import t


# Adapted from https://github.com/fzi-forschungszentrum-informatik/TSInterpret/blob/main/TSInterpret/InterpretabilityModels/Saliency/Saliency_Base.py
def plot_saliency(
    ts_data,
    saliency_list,
    perturbed=None,
    feature_names=None,
    figsize=None,
    suptitle=None,
):
    if feature_names is not None:
        assert len(feature_names) == len(
            ts_data
        ), "The length of feature names and data is not the same"

    plt.style.use("default")

    if figsize is None:
        figsize = (10, len(ts_data) * 1.2)

    fig, axn = plt.subplots(len(ts_data), 1, sharex=True, sharey=True, figsize=figsize)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    for i, channel in enumerate(ts_data):
        axn012 = axn[i].twinx()

        # Plot saliency
        sns.heatmap(
            saliency_list[i].reshape(1, -1),
            fmt="g",
            cmap="viridis",
            cbar=i == 0,
            cbar_ax=None if i else cbar_ax,
            cbar_kws={"label": "Saliency / Mask intensity"},
            ax=axn[i],
            yticklabels=False,
            vmin=0,
            vmax=1,
        )

        xs = range(1, len(channel.reshape(-1)) + 1)

        if perturbed is not None:
            # Plot learned perturbations
            axn012.plot(
                xs,
                perturbed[i].detach().numpy(),
                color="red",
                label="Perturbed data",
                marker=".",
            )

        # Plot original data
        axn012.plot(
            xs, channel.flatten(), color="#91BBDE", label="Original data", marker="."
        )

        plt.xlabel("Time", fontweight="bold", fontsize="large")
        feat_name = feature_names[i] if feature_names is not None else f"Feature {i}"
        plt.ylabel(feat_name, fontweight="bold", fontsize="medium")
    plt.legend()
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    fig.suptitle(suptitle, y=1.01)
    plt.show()


# Adapted from appendix of https://arxiv.org/abs/2106.05303
def plot_heatmap(
    saliencies,
    figsize=None,
    title: str = "Mask coefficients over time",
    subtitles: list or None = None,
    cbar_title: str = "Saliency / Mask intensity",
):
    plt.style.use("default")
    N = len(saliencies)
    if figsize is None:
        figsize = (10, N * 2)

    fig, axn = plt.subplots(int(np.ceil(N / 2)), 2, figsize=figsize)
    axn = axn.flatten()
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])

    color_map = sns.diverging_palette(10, 133, as_cmap=True)

    for idx, (method, saliency) in enumerate(saliencies.items()):
        sns.heatmap(
            saliency,
            cmap=color_map,
            cbar=idx == 0,
            cbar_ax=None if idx else cbar_ax,
            cbar_kws={"label": cbar_title},
            ax=axn[idx],
            yticklabels=True,
            linecolor="#d8cbd5",
            linewidths=0.4,
            vmin=0,
            vmax=1,
        )
        axn[idx].tick_params(axis="both", which="major", labelsize=7)
        if subtitles is not None:
            axn[idx].set_title(subtitles[idx])
        else:
            axn[idx].set_title(f"Method: {method}")
        axn[idx].set_xlabel("Time")
        axn[idx].set_ylabel("Feature number")

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_topk_heatmap(
    saliencies,
    subtitles: list or None = None,
    figsize=None,
):
    plt.style.use("default")
    N = len(saliencies)
    if figsize is None:
        figsize = (10, N * 2)

    fig, axn = plt.subplots(int(np.ceil(N / 2)), 2, figsize=figsize)
    axn = axn.flatten()
    color_map = sns.diverging_palette(10, 133, as_cmap=True)

    legend_elements = [
        Line2D(
            [0],
            [0],
            color="w",
            marker="s",
            markerfacecolor="#398649",
            label="Salient",
            markersize=15,
        ),
        Line2D(
            [0],
            [0],
            color="w",
            marker="s",
            markerfacecolor="#DA3B46",
            label="Non-salient",
            markersize=15,
        ),
    ]

    for idx, (method, saliency) in enumerate(saliencies.items()):
        sns.heatmap(
            saliency,
            cmap=color_map,
            cbar=False,
            ax=axn[idx],
            yticklabels=True,
            linecolor="#d8cbd5",
            linewidths=0.4,
            vmin=0,
            vmax=1,
        )
        axn[idx].tick_params(axis="both", which="major", labelsize=7)
        if subtitles:
            axn[idx].set_title(method)
        else:
            axn[idx].set_title(f"Method: {method}")
        axn[idx].set_xlabel("Time")
        axn[idx].set_ylabel("Feature number")

    fig.tight_layout(rect=[0, 0, 0.92, 1])
    fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(0.9, 0.5),
        bbox_transform=fig.transFigure,
    )
    plt.show()


def plot_components(
    orig, perturbed, mask, perturbation, suptitle=None, figsize=(10, 6)
):
    plt.style.use("ggplot")
    plt.rcParams.update({"axes.titlesize": "medium"})

    fig, axs = plt.subplots(5, 1, figsize=figsize)

    orig = orig.flatten().numpy()
    perturbed = perturbed.detach().numpy()
    mask = mask.detach().numpy()
    perturbation = perturbation.detach().numpy()

    xs = range(0, len(orig.reshape(-1)))
    axs[0].plot(xs, mask, marker=".")
    axs[0].set_title("Learned mask/saliency: m")

    axs[1].plot(xs, perturbation, marker=".")
    axs[1].set_title(
        "Learned perturbation: NN(x)",
    )

    axs[2].plot(xs, mask * orig, marker=".")
    axs[2].set_title("m * x")

    axs[3].plot(xs, (1 - mask) * perturbation, marker=".")
    axs[3].set_title("(1 - m) * NN(x)")

    axs[4].plot(xs, perturbed, label="m * x + (1 - m) * NN(x)", marker=".")
    axs[4].plot(xs, orig, label="Original data", marker=".")
    axs[4].set_title("Perturbed vs Original data")
    axs[4].legend(fontsize=7)

    plt.xlabel("Time", fontweight="bold")
    plt.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def plot_mean_attributions(
    attr, alpha=0.95, title=None, xtick_labels=None, debug=False
):
    means_over_time = []
    # Compute means per time series (or feature)
    for patient in attr:
        means_over_time.append(patient.mean(axis=0))

    means_over_time = np.array(means_over_time)

    # Compute means and confidence intervals
    feature_means = []
    feature_err = []
    for feature in means_over_time.T:
        mean = feature.mean()
        feature_means.append(mean)
        conf_interval = t.interval(
            confidence=alpha,
            df=len(feature) - 1,
            loc=np.mean(feature),
            scale=np.std(feature) / np.sqrt(len(feature)),
        )
        # This is needed for the plotting function only, it will always subtract
        # and add these confidence intervals to the mean
        conf_interval = (mean - conf_interval[0], conf_interval[1] - mean)
        feature_err.append(conf_interval)

    feature_means = np.array(feature_means)
    feature_err = np.array(feature_err)
    if debug:
        # feature_err = th.zeros_like(th.tensor(feature_err))
        # Top k feature importances
        k = 5
        ind = np.argpartition(feature_means, -k)[-k:].astype(int)
        print(ind)
        print(np.array(xtick_labels)[ind])

    xs = np.arange(len(feature_means))
    plt.errorbar(
        xs, feature_means, yerr=feature_err.T, fmt="^", label="Mean attribution"
    )
    plt.ylabel("Attribution")
    plt.xticks(ticks=xs, labels=xtick_labels, rotation=50, fontsize=8, ha="right")
    plt.legend()
    plt.title(title)
    plt.show()
