import argparse
import os

import random

import numpy as np
import torch
from train_torch import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from sklearn.decomposition import PCA

def sample(model, slider_vals, pca):
    """
    Function that generates an image given a latent dimension.
    Inputs:
        model - The VAE model that is currently being trained.
        slider_vals - the 20 values given by the slider values
        pca - PCA object, to get back the latent dimension
    """

    z = pca.inverse_transform(slider_vals) #create latent dimension from pca values
    z = torch.FloatTensor(z)
    samples = model.generate_from_z(z) #only create 1 image

    B, P, H, W, C = samples.size()

    sampled = np.zeros((H, W * 3, 3))
    for p in range(P):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    sampled[h, w + p * 200, c] = samples[0,p,h,w,c] / 2 + 0.5

    return sampled

def calculate_pca(model, test_loader):
    """Returns a pca object of the latent space of the test loader"""
    total_loss, total_rec_loss, total_reg_loss = 0., 0., 0.
    num_samples = 0
    tot_mean = []

    for batch_ndx, sample in enumerate(test_loader):
        model.eval()
        input = sample.to(model.device)
        mean, _ = model.get_z(sample) #Only look at the mean, ignore the std
        tot_mean = tot_mean + mean.tolist()

    pca = PCA(n_components=20)
    pca.fit(tot_mean)

    return pca

def visualize(args):
    # The function to be called when clicked on update
    def update(val):
        slider_vals = []
        for i in range(20):
            slider_vals.append(sliders[i].val)
        image.imshow(sample(model, slider_vals, pca), interpolation='nearest')
        fig.canvas.draw_idle()

    # The function to be called when clicked on random
    def random_vals(val):
        slider_vals = []
        for i in range(20):
            sliders[i].set_val(random.random()*4-2)
            slider_vals.append(sliders[i].val)
        image.imshow(sample(model, slider_vals, pca), interpolation='nearest')
        fig.canvas.draw_idle()

    # The function to be called when clicked on reset
    def reset(val):
        slider_vals = []
        for i in range(20):
            sliders[i].set_val(0)
            slider_vals.append(sliders[i].val)
        image.imshow(sample(model, slider_vals, pca), interpolation='nearest')
        fig.canvas.draw_idle()

    #Load model
    model = VAE(z_dim=args.z_dim, num_blocks=args.num_blocks, c_hidden=args.c_hidden,
                beta=args.beta, lr=args.lr)
    checkpoint_path = args.model_file
    model.load_state_dict(torch.load(checkpoint_path))

    #Calculate PCA
    train_loader, val_loader, test_loader = prepare_data_loader(batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = (test_loader if args.no_progress_bar else
                    tqdm(test_loader, desc="Calculating PCA", leave=False))
    pca = calculate_pca(model, test_loader)

    #Create image when all sliders are set to 0
    slider_vals = torch.zeros(20)
    sampled = sample(model, slider_vals, pca)

    fig, ax = plt.subplots()
    plt.axis('off')

    image = plt.axes([0,0.5,1,0.5])
    image.imshow(sampled, interpolation='nearest')
    image.axis('off')

    sliders = []
    for i in range(20):
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
    button_update = Button(updateax, 'Update', hovercolor='0.975')

    randomax = plt.axes([0.6, 0.45, 0.1, 0.04])
    button_random = Button(randomax, 'Random', hovercolor='0.975')

    resetax = plt.axes([0.4, 0.45, 0.1, 0.04])
    button_reset = Button(resetax, 'Reset', hovercolor='0.975')

    button_update.on_clicked(update)
    button_random.on_clicked(random_vals)
    button_reset.on_clicked(reset)

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--z_dim', default=1024, type=int,
                        help='Dimensionality of latent space')
    parser.add_argument('--num_blocks', default=2, type=int,
                        help='Number of blocks in each set of equal sized convolutions')
    parser.add_argument('--c_hidden', default=6, type=int,
                        help='Number of channels in convolutions')

    # Other hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--beta', default=0.0001, type=float,
                        help='Beta')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--model_file', default='results/model.pt', type=str,
                        help='File containing model to process.')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--no_progress_bar', action='store_true',
                        help=('Do not use a progress bar indicator for interactive experimentation. '))


    args = parser.parse_args()

    visualize(args)
