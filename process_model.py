import argparse
import os

import numpy as np
import torch
from train_torch import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

image = None
model = None
z = None
fig = None
sliders = []

# The function to be called anytime a slider's value changes
def update(val):
    z = torch.zeros(args.z_dim)
    for i in range(20):
        z[i] = sliders[i].val
    image.imshow(sample(model, z), interpolation='nearest')
    fig.canvas.draw_idle()

def sample(model, z):
    """
    Function that generates samples given a latent dimension.
    Inputs:
        model - The VAE model that is currently being trained.
        z - latent dimension as a 1d tensor of size z_dim
    """

    samples = model.generate_from_z(z) #only create 1 image

    B, P, H, W, C = samples.size()

    sampled = np.zeros((H, W * 3, 3))
    for p in range(P):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    sampled[h, w + p * 200, c] = samples[0,p,h,w,c] / 2 + 0.5

    return sampled

def visualize(args):
    # Create model
    global model
    global fig
    global image
    global sliders

    model = VAE(z_dim=args.z_dim, num_blocks=args.num_blocks, c_hidden=args.c_hidden,
                beta=args.beta, lr=args.lr)
    checkpoint_path = os.path.join(args.experiment_dir, 'checkpoints', 'model.pt')
    model.load_state_dict(torch.load(checkpoint_path))

    z = torch.zeros(args.z_dim)
    sampled = sample(model, z)

    fig, ax = plt.subplots()
    plt.axis('off')

    image = plt.axes([0,0.5,1,0.5])
    image.imshow(sampled, interpolation='nearest')
    image.axis('off')

    for i in range(20):
        print([0.025 + i % 10, 0.25 - 0.2 * int(i / 10), 0.05, 0.15])
        axamp = plt.axes([0.025 + 0.1 * (i % 10), 0.25 - 0.2 * int(i / 10), 0.05, 0.12])
        sliders.append(Slider(
            ax=axamp,
            label=i,
            valmin=-2,
            valmax=2,
            valinit=0,
            orientation="vertical"
        ))

    updateax = plt.axes([0.8, 0.45, 0.1, 0.04])
    button = Button(updateax, 'Reset', hovercolor='0.975')

    button.on_clicked(update)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=750, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_blocks', default=2, type=int,
                        help='Number of blocks in each set of equal sized convolutions')
    parser.add_argument('--c_hidden', default=4, type=int,
                        help='Number of channels in convolutions')

    # Other hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--beta', default=0.0001, type=float,
                        help='Beta')
    parser.add_argument('--experiment_dir', default='', type=str,
                        help='Directory containing model to continue training from.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')

    args = parser.parse_args()

    visualize(args)
