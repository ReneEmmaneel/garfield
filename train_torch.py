import argparse
import os
import datetime
import statistics
import random

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

from data_handler import *
from cnn_encoder_decoder import CNNEncoder, CNNDecoder
from utils import *

import json

class VAE(nn.Module):

    def __init__(self, z_dim, num_blocks, c_hidden, beta, *args,
                 **kwargs):
        """
        PyTorch module that summarizes all components to train a VAE.
        Inputs:
            z_dim - Dimensionality of latent space
            num_blocks - number of blocks per resnet layer of same size
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
        num_conv_blocks - number of resnet layers of different sizes
        """
        super().__init__()
        self.z_dim = z_dim

        self.beta = beta

        num_conv_blocks = 4

        self.encoder = CNNEncoder(z_dim=z_dim, num_blocks=[num_blocks]*num_conv_blocks, c_hidden=[c_hidden*2**i for i in range(num_conv_blocks)])
        self.decoder = CNNDecoder(z_dim=z_dim, num_blocks=[num_blocks]*num_conv_blocks, c_hidden=[c_hidden*2**i for i in range(num_conv_blocks)][::-1])

    def forward(self, imgs, epoch):
        """
        The forward function calculates the VAE loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W].
                   The input images are converted to 4-bit, i.e. integers between 0 and 15.
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            loss - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """
        imgs = imgs / 255 * 2.0 - 1.0  # Move images between -1 and 1
        mean, std = self.encoder(imgs)

        # Clamp mean and std, this ensures values don't explode
        mean = torch.clamp(mean, min=-10, max=10)
        std = torch.clamp(std, min=-8, max=3)

        z = sample_reparameterize(mean, torch.exp(std))
        out = self.decoder(z)

        L_rec = F.mse_loss(out, imgs)
        L_reg = torch.mean(KLD(mean, std))

        #Count Regularization loss more the longer the training goes on
        beta = self.beta
        if epoch >= 50: beta *= 4
        if epoch >= 150: beta *= 4

        loss = L_rec + beta * L_reg

        return L_rec, L_reg, loss

    @torch.no_grad()
    def sample(self, batch_size):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled images. Shape: [B,P,H,W,C]
        """
        z = torch.normal(mean=torch.zeros(batch_size, self.z_dim), std=1).to(self.device)
        x_samples = self.decoder(z)
        return x_samples

    @torch.no_grad()
    def generate_from_z(self, z):
        """
        Function for generating an output image from latent space z.
        Inputs:
            z - latent space of size [z_dim] or [B,z_dim] as tensor.
        Outputs:
            x_samples - Sampled, 4-bit images. Shape: [B,P,H,W,C]
        """
        if len(z.size()) == 1:
            z = z.reshape(1, z.size()[0])
        z = z.to(self.device)
        x_samples = self.decoder(z)
        return x_samples

    @torch.no_grad()
    def get_z(self, imgs):
        imgs = imgs / 255 * 2.0 - 1.0  # Move images between -1 and 1
        mean, std = self.encoder(imgs)
        return mean, std

    @torch.no_grad()
    def reconstruct_from_image(self, imgs):
        imgs = imgs / 255 * 2.0 - 1.0  # Move images between -1 and 1
        mean, std = self.encoder(imgs)
        z = sample_reparameterize(mean, torch.exp(std))
        out = self.decoder(z)

        return out

    @property
    def device(self):
        """
        Property function to get the device on which the model is.
        """
        return self.decoder.device


def sample_and_save(model, epoch, summary_writer, log_dir, batch_size=1):
    """
    Function that generates and saves samples from the VAE.  The generated
    samples and mean images should be saved, and can eventually be added to a
    TensorBoard logger if wanted.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        log_dir - Directory to log the created images
        batch_size - Number of images to generate/sample
    """

    samples = model.sample(batch_size)

    B, P, H, W, C = samples.size()

    sampled = torch.zeros(B*3, 3, H, W)
    for b in range(B):
        for p in range(P):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        sampled[b*3+p,c,h,w] = samples[b,p,h,w,c] / 2 + 0.5

    grid = make_grid(sampled, nrow=3)
    path = '{}/samples_epoch{}.jpg'.format(log_dir, epoch)
    save_image(grid, path)

    if summary_writer:
        summary_writer.add_image("Samples after epoch {}".format(epoch), grid)

def plot_random_reconstruction(model, epoch, summary_writer, log_dir, loader, loader_name, batch_size=2):
    """
    Function that plots an image from the loader, and plots the image after
    reconstructing it using the model.
    Inputs:
        model - The VAE model that is currently being trained.
        epoch - The epoch number to use for TensorBoard logging and saving of the files.
        summary_writer - A TensorBoard summary writer to log the image samples.
        log_dir - Directory to log the created images
        loader - The dataloader used to sample images from
        loader_name - The name (used to save the images, usually 'train' or 'validation')
    """

    imgs = next(iter(loader))[0:batch_size]
    reconstruction = model.reconstruct_from_image(imgs)
    B, P, H, W, C = imgs.size()

    sampled = torch.zeros(batch_size * 6, 3, H, W)
    for b in range(B):
        for p in range(P):
            for h in range(H):
                for w in range(W):
                    for c in range(C):
                        sampled[b*6+p,c,h,w] = imgs[b,p,h,w,c] / 255 #Between 0 and 1
                        sampled[b*6+p+3,c,h,w] = reconstruction[b,p,h,w,c] / 2 + 0.5 #Between 0 and 1

    grid = make_grid(sampled, nrow=3)
    path = '{}/reconstruction_{}_epoch{}.jpg'.format(log_dir, loader_name, epoch)
    save_image(grid, path)

    if summary_writer:
        summary_writer.add_image("Reconstruction from {} set after epoch {}".format(loader_name, epoch), grid)


@torch.no_grad()
def test_vae(model, data_loader):
    """
    Function for testing a model on a dataset.
    Inputs:
        model - VAE model to test
        data_loader - Data Loader for the dataset you want to test on.
    Outputs:
        average_loss - Average loss
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """
    total_loss, total_rec_loss, total_reg_loss = 0., 0., 0.
    num_samples = 0
    for batch_ndx, sample in enumerate(data_loader):
        model.eval()
        input = sample.to(model.device)
        with torch.no_grad():
            L_rec, L_reg, loss = model(input, 1000) #Epoch is set to 1000 to test with highest beta value

        batch_size = len(sample[0])

        total_loss += loss * batch_size
        total_rec_loss += L_rec * batch_size
        total_reg_loss += L_reg * batch_size

        num_samples += batch_size

    average_loss = total_loss / num_samples
    average_rec_loss = total_rec_loss / num_samples
    average_reg_loss = total_reg_loss / num_samples
    return average_loss, average_rec_loss, average_reg_loss


def train_vae(model, train_loader, optimizer, epoch):
    """
    Function for training a model on a dataset. Train the model for one epoch.
    Inputs:
        model - VAE model to train
        train_loader - Data Loader for the dataset you want to train on
        optimizer - The optimizer used to update the parameters
        epoch - Current epoch, used to set the beta value
    Outputs:
        average_loss - Average loss
        average_rec_loss - Average reconstruction loss
        average_reg_loss - Average regularization loss
    """

    total_loss, total_rec_loss, total_reg_loss = 0., 0., 0.
    num_samples = 0
    for batch_ndx, sample in enumerate(train_loader):
        model.train()
        input = sample.to(model.device)
        optimizer.zero_grad()
        L_rec, L_reg, loss = model(input, epoch)
        loss.backward()
        optimizer.step()

        batch_size = len(sample[0])

        total_loss += loss * batch_size
        total_rec_loss += L_rec * batch_size
        total_reg_loss += L_reg * batch_size

        num_samples += batch_size

    average_loss = total_loss / num_samples
    average_rec_loss = total_rec_loss / num_samples
    average_reg_loss = total_reg_loss / num_samples
    return average_loss, average_rec_loss, average_reg_loss

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    """
    Main Function for the full training & evaluation loop of a VAE model.
    Make use of a separate train function and a test function for both
    validation and testing (testing only once after training).
    Inputs:
        args - Namespace object from the argument parser
    """
    if args.seed is not None:
        seed_everything(args.seed)

    if args.continue_from:
        experiment_dir = args.continue_from
        print(f'Continuing from {experiment_dir}\nWarning: summary writer is not used when continue training')
        summary_writer = None
        #Note: summary writer does not work when continuing training
    else:
        experiment_dir = os.path.join(args.log_dir, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        summary_writer = SummaryWriter(experiment_dir)

    # Prepare logging
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    samples_dir = os.path.join(experiment_dir, 'samples')
    reconstruction_train_dir = os.path.join(experiment_dir, 'reconstruction_train')
    reconstruction_val_dir = os.path.join(experiment_dir, 'reconstruction_validation')
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(reconstruction_train_dir, exist_ok=True)
    os.makedirs(reconstruction_val_dir, exist_ok=True)
    if not args.progress_bar:
        print("[INFO] The progress bar has been suppressed. For updates on the training " + \
              f"progress, check the TensorBoard file at {experiment_dir}. If you " + \
              "want to see the progress bar, use the argparse option \"progress_bar\".\n")

    # Load dataset
    train_loader, val_loader, test_loader = prepare_data_loader(batch_size=args.batch_size, num_workers=args.num_workers)

    # Create model
    model = VAE(z_dim=args.z_dim,
                num_blocks=args.num_blocks,
                c_hidden=args.c_hidden,
                beta=args.beta,
                lr=args.lr)

    if args.continue_from:
        checkpoint_path = os.path.join(experiment_dir, 'checkpoints', 'model.pt')
        model.load_state_dict(torch.load(checkpoint_path))

        # Getting previous variables for finding best model
        with open(os.path.join(checkpoint_dir, "model_info.txt")) as json_file:
            data = json.load(json_file)
            best_epoch_idx = data['epoch']
            best_val_loss = data['loss']
    else:
        best_val_loss = float('inf')
        best_epoch_idx = 0

    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{'Loaded' if args.continue_from else 'Created'} model with {num_parameters} parameters")

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Using device {device}")
    epoch_iterator = (trange(1 + best_epoch_idx, args.epochs + 1 + best_epoch_idx, desc=f"VAE")
                      if args.progress_bar else range(1 + best_epoch_idx, args.epochs + 1 + best_epoch_idx))
    for epoch in epoch_iterator:
        # Sample image grid before training starts
        if epoch == 1:
            plot_random_reconstruction(model, 0, summary_writer, reconstruction_train_dir, train_loader, 'train')
            plot_random_reconstruction(model, 0, summary_writer, reconstruction_val_dir, val_loader, 'validation')
            sample_and_save(model, 0, summary_writer, samples_dir, 1)

        # Training epoch
        train_iterator = (tqdm(train_loader, desc="Training", leave=False)
                          if args.progress_bar else train_loader)
        epoch_train_loss, train_rec_loss, train_reg_loss = train_vae(
            model, train_iterator, optimizer, epoch)

        # Validation epoch
        val_iterator = (tqdm(val_loader, desc="Testing", leave=False)
                        if args.progress_bar else val_loader)
        epoch_val_loss, val_rec_loss, val_reg_loss = test_vae(model, val_iterator)

        # Logging to TensorBoard
        if summary_writer:
            summary_writer.add_scalars(
                "loss", {"train": epoch_train_loss, "val": epoch_val_loss}, epoch)
            summary_writer.add_scalars(
                "Reconstruction Loss", {"train": train_rec_loss, "val": val_rec_loss}, epoch)
            summary_writer.add_scalars(
                "Regularization Loss", {"train": train_reg_loss, "val": val_reg_loss}, epoch)

        sample_and_save(model, epoch, summary_writer, samples_dir, 3)
        plot_random_reconstruction(model, epoch, summary_writer, reconstruction_train_dir, train_loader, 'train')
        plot_random_reconstruction(model, epoch, summary_writer, reconstruction_val_dir, val_loader, 'validation')

        # Saving best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch_idx = epoch
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))

            data = {'epoch': epoch, 'loss': epoch_val_loss.item()}
            with open(os.path.join(checkpoint_dir, "model_info.txt"), 'w') as outfile:
                json.dump(data, outfile)

    # Load best model for test
    print(f"Best epoch: {best_epoch_idx}. Load model for testing.")
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "model.pt")))

    # Test epoch
    test_loader = (tqdm(test_loader, desc="Testing", leave=False)
                   if args.progress_bar else test_loader)
    test_loss, _, _ = test_vae(model, test_loader)
    print(f"Test loss: {test_loss}")
    if summary_writer:
        summary_writer.add_scalars("loss", {"test": test_loss}, best_epoch_idx)

    return test_loss


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

    # Optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')
    parser.add_argument('--beta', default=0.01, type=float,
                        help='Beta')

    # Other hyperparameters
    parser.add_argument('--data_dir', default='data/all_images/', type=str,
                        help='Directory where to look for the data.')
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' + \
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '))
    parser.add_argument('--log_dir', default='logs', type=str,
                        help='Directory where the PyTorch logs should be created.')
    parser.add_argument('--continue_from', default='', type=str,
                        help='Directory containing model to continue training from.')

    args = parser.parse_args()

    main(args)
