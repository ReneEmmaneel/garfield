import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, upsample=False, c_out=-1):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        if not subsample and not upsample:
            c_out = c_in

        #The first convolutional layer in the module can be used to upsample or downsample
        if subsample:
            first_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=2, bias=False)
        elif upsample:
            first_conv = nn.ConvTranspose2d(c_in, c_out, kernel_size=3, output_padding=1, padding=1, stride=2, bias=False)
        else:
            first_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=1, bias=False)

        #The path to calculate z
        self.net = nn.Sequential(
            first_conv,
            nn.BatchNorm2d(c_out),
            act_fn(),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out)
        )

        # 1x1 convolution with stride 2 means we take the upper left value, and transform it to new output size
        self.downsample = nn.Conv2d(c_in, c_out, kernel_size=1, stride=2) if subsample else None #decrease the size by 2x
        # Upsampling, using the simple 'nearest' mode
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), #increase the size by 2x
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=1) #Remove half the channels
        ) if upsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        elif self.upsample is not None:
            x = self.upsample(x)

        out = z + x
        out = self.act_fn(out)
        return out

class CNNEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 3,
                 z_dim: int = 20, num_blocks=[6,6,6], c_hidden=[16,32,64],
                 last_conv_size=(25,22)):
        """Encoder with a CNN network

        Inputs:
            num_input_channels - Number of input channels of the image.
                                 Garfield has 3 color channels.
            z_dim - Dimensionality of latent representation z
            num_blocks - List with the number of ResNet blocks to use. The first block of each group uses downsampling.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually multiplied by 2 the deeper we go.
            last_conv_size - The size of the last convolutional layer (height x width)
            to_input_size - First transform image to this size, to easily downsample
        """
        super().__init__()
        act_fn = nn.ReLU

        self.input_net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hidden[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_hidden[0]),
            act_fn()
        )

        blocks = []
        for block_idx, block_count in enumerate(num_blocks):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0) # Subsample the first block of each group, except the very first one.
                blocks.append(
                    ResNetBlock(c_in=c_hidden[block_idx if not subsample else (block_idx-1)],
                                act_fn=act_fn,
                                subsample=subsample,
                                c_out=c_hidden[block_idx])
                )
        self.net = nn.Sequential(*blocks)

        self.flatten_net = nn.Sequential(
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(last_conv_size[0] * last_conv_size[1] * c_hidden[-1], 2*z_dim),
            nn.BatchNorm1d(2*z_dim),
            act_fn()
        )

        self.conv_net = nn.Sequential(
            self.input_net,
            self.net,
            self.flatten_net
        )

        self.combine_layer = nn.Sequential(
            nn.Linear(6*z_dim, 2*z_dim)
        )

    def forward(self, x):
        """
        Inputs:
            x - Input batch with images of shape [B,P,H,W,C]
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """
        x = x.float()

        x = x.permute(0,1,4,2,3) #Permute to [B,P,C,H,W]
        p1, p2, p3 = torch.chunk(x, 3, dim=1)
        p1 = self.conv_net(torch.squeeze(p1)) #Shape is [B,C,H,W] -> [2*z_dim]
        p2 = self.conv_net(torch.squeeze(p2))
        p3 = self.conv_net(torch.squeeze(p3))

        x = torch.cat((p1, p2, p3), dim=1)
        x = self.combine_layer(x)

        x = torch.chunk(x, 2, dim=-1)
        return x[0], x[1]

class MakeRectangle(nn.Module):
    def __init__(self, size=(25,22)):
        super().__init__()
        self.size = size

    def forward(self, x):
        return x.reshape(x.shape[0], -1, self.size[1], self.size[0])

class CNNDecoder(nn.Module):
    def __init__(self, num_input_channels: int = 3,
                 z_dim: int = 20, num_blocks=[6,6,6], c_hidden=[64,32,16],
                 first_conv_size=(25,22)):
        """Decoder with a CNN network.

        Inputs:
            num_input_channels- Number of channels of the image
            z_dim - Dimensionality of latent representation z
            num_blocks - List with the number of ResNet blocks to use. The last block of each group, except last, uses upsampling.
            c_hidden - List with the hidden dimensionalities in the different blocks. Usually divided by 2 the deeper we go.
            first_conv_size - The size of the first convolutional layer (width by heights)
        """
        super().__init__()
        act_fn = nn.ReLU

        self.input_net = nn.Sequential(
            nn.Linear(z_dim, first_conv_size[0] * first_conv_size[1] * c_hidden[0]),
            MakeRectangle(first_conv_size),
            act_fn()
        )

        blocks = []
        for block_idx, block_count in enumerate(num_blocks):
            for bc in range(block_count):
                upsample = (bc == 0 and block_idx > 0) # Upsample the last block of each group, except for the first one
                blocks.append(
                    ResNetBlock(c_in=c_hidden[block_idx if not upsample else (block_idx-1)],
                                act_fn=act_fn,
                                upsample=upsample,
                                c_out=c_hidden[block_idx])
                )
        self.net = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
            nn.Conv2d(c_hidden[-1], num_input_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

        self.conv_net = nn.Sequential(
            self.input_net,
            self.net,
            self.output_net
        )

        self.spread_layer = nn.Sequential(
            nn.Linear(z_dim, 3*z_dim)
        )

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z.
                This should be a logit output *without* a sigmoid applied on it.
                Shape: [B,num_input_channels,28,28]
        """
        x = self.spread_layer(z)

        #Create all 3 images, and add a dummy dimension (for panel dimension)
        p1, p2, p3 = torch.chunk(x, 3, dim=-1)
        p1 = self.conv_net(p1)[:, None]
        p2 = self.conv_net(p2)[:, None]
        p3 = self.conv_net(p3)[:, None]

        #Stack the output images, reorder from [B,P,C,H,W] to [B,P,H,W,C]
        x = torch.cat((p1, p2, p3), dim=1)
        x = x.permute(0,1,3,4,2)

        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device
