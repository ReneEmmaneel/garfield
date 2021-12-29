import torch
from torchvision.utils import make_grid
from torch import distributions
import numpy as np


def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std.
            The tensor should have the same shape as the mean and std input tensors.
    """
    assert not (std < 0).any().item(), "The reparameterization trick got a negative std as input. " + \
                                       "Are you sure your input is std and not log_std?"

    z = torch.normal(mean=torch.zeros(mean.shape), std=1).to(mean.device)
    z = z * std + mean
    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    See Section 1.3 for the formula.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        KLD - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    exp_squared = torch.exp(log_std)**2
    KLD = (exp_squared + mean**2 - 1 - torch.log(exp_squared))/2
    return torch.sum(KLD, dim=-1)


def elbo_to_bpd(elbo, img_shape):
    """
    #NOTE: note used at the moment

    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        img_shape - Shape of the input images, representing [batch, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """
    num_dimensions = float(torch.prod(torch.as_tensor(img_shape[1:])))
    bpd = elbo * np.log2(np.e) * num_dimensions**-1
    return bpd
