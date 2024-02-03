def get_perturbed_data(mask, original, noise):
    return mask * original + (1-mask)*noise