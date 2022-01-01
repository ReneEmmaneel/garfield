# Garfield Variational Autoencoder (VAE)

A VAE is an autoencoder, with the goal of reconstructing an input image while first pushing it through a bottleneck (latent space)
In this repository the garfield strips are used as the dataset.
There is functionality to download the dataset, to train the dataset and to use the trained model for generating new garfield strips.

## The structure

The VAE consist of an encoder and decoder.
The encoder learns a distribution of the input image of a given dimensionality.
This distribution has a mean and standard deviation for each dimension.
The decoder takes a sample from this learned distribution, and tries to reconstruct the original image as good as possible.

For both the encoder and decoder we use three convolution networks, for each panel. 
The weights between the networks are shared. 
The networks consist of ResNetBlocks with upsampling/downsampling. 
The networks are connected to the latent space through a dense layer.

The loss function used is L = MeanSquaredError(input_image, output_image) + beta * KL(latent distribution, normal distribution).
Here, KL is the Kullback–Leibler divergence where a lower value means the latent distribution is more similiar to the normal distribution.
The KL divergence is used to prevent overfitting: otherwise each input image would just map to an unique point in the latent space instead of learning features.
The beta value is a hyperparameter.

## Usage

`train_torch.py` will download the garfield dataset (this can take a while) and then train the model using the given parameters.
The command used to train is as follows:

`python3 train_torch.py --beta 0.0001 --z_dim 1024 --batch_size 64 --epochs 400 --progress_bar --num_workers 0 --num_blocks 2 --c_hidden 6`

To use a trained model for generating new strips use `process_model.py`. It can take a while to calculate the PCA. 
Note that you need to add the model parameters as used when training the model.

`python process_model.py --experiment_dir logs/[dir_name]/ --z_dim 1024 --c_hidden 6 --progress_bar`
