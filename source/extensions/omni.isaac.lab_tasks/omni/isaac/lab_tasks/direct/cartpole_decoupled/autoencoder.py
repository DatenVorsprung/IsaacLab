import argparse
import os
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, input_shape: tuple,
                 latent_dim: int, act_fn: object = nn.GELU):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers.
                Deeper layers might use a duplicate of it.
           input_shape: (Width, Height) of the input images
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.c_hid = base_channel_size
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_input_channels = num_input_channels
        self.conv_net = nn.Sequential(
            nn.Conv2d(num_input_channels, self.c_hid, kernel_size=8, stride=4),
            act_fn(),
            nn.Conv2d(self.c_hid, 2*self.c_hid, kernel_size=4, stride=2),
            act_fn(),
            nn.Conv2d(2*self.c_hid, 2*self.c_hid, kernel_size=3, stride=1),
            act_fn(),
        )
        self.__post_init__()

    def __post_init__(self):
        """ Initialize the output head """
        hidden_shape = self._get_hidden_shape()
        # take the last two dimensions of the hidden shape to get the dimension after flattening
        hidden_dim = 2 * self.c_hid * torch.tensor(hidden_shape).prod().item()
        self.head = nn.Linear(hidden_dim, self.latent_dim)

    def _get_hidden_shape(self):
        """ get shape of feature vector after the convolutions before flattening """
        dummy_input = torch.rand(1, self.num_input_channels, *self.input_shape)
        return self.conv_net(dummy_input).shape[-2:]

    def forward(self, x):
        x = self.conv_net(x)
        x = nn.Flatten(start_dim=1, end_dim=-1)(x)
        return self.head(x)


class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, hidden_shape: tuple,
                 latent_dim: int, act_fn: object = nn.GELU):
        """Decoder.

        Args:
           num_input_channels : Number of channels of the image to reconstruct. 
           base_channel_size : Number of channels we use in the last convolutional layers.
                Early layers might use a duplicate of it.
           hidden_shape: Shape of the hidden dimension after first linear layer
           latent_dim : Dimensionality of latent representation z
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.c_hid = base_channel_size
        self.hidden_shape = hidden_shape
        self.latent_dim = latent_dim
        self.act_fn = act_fn
        self.transpose_conv_net = nn.Sequential(
            nn.ConvTranspose2d(2 * self.c_hid, 2 * self.c_hid, kernel_size=3, stride=1),
            act_fn(),
            nn.ConvTranspose2d(2 * self.c_hid, self.c_hid, kernel_size=4, stride=2, output_padding=1),
            act_fn(),
            nn.ConvTranspose2d(self.c_hid, num_input_channels, kernel_size=8, stride=4, padding=1,
                               output_padding=2),
            nn.Tanh(),
        )
        self.__post_init__()

    def __post_init__(self):
        """ Initialize the first linear layer of the decoder """
        num_hidden_dim = 2 * self.c_hid * torch.tensor(self.hidden_shape).prod().item()
        self.linear = nn.Sequential(nn.Linear(self.latent_dim, num_hidden_dim), self.act_fn())

    def forward(self, x):
        x = self.linear(x)
        # reshaping to hidden_shape
        x = x.reshape(x.shape[0], -1, *self.hidden_shape)
        x = self.transpose_conv_net(x)
        return x


class Autoencoder(L.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        input_shape: tuple,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3
    ):
        super().__init__()
        self.input_shape = input_shape
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, input_shape, latent_dim)
        hidden_shape = self.encoder._get_hidden_shape()
        self.decoder = decoder_class(num_input_channels, base_channel_size, hidden_shape,
                                     latent_dim)
        # Example input array needed for visualizing the graph of the network
        # pytorch default channel ordering is (C,H,W)
        self.example_input_array = torch.zeros(2, num_input_channels, *input_shape)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # if the reconstructed image is bigger than the original -> slice it
        # TODO This is a naive approach that assumes the reconstructed image is large than the
        # original; we could do something more elaborate here
        if x_hat.shape[-2:] != self.input_shape:
            x_hat = x_hat[:, :, :self.input_shape[0], :self.input_shape[1]]
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = F.mse_loss(x, x_hat, reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)



