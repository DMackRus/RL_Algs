import torch.nn as nn

from utils import (
    initialize_weights,
    horizontal_forward,
    create_normal_dist,
)

class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()

        self.config = config.parameters.dreamer.encoder

        activation = getattr(nn, self.config.activation)()

        self.observation_shape = observation_shape

        # NOTE:
        # Set encoder.depth = 16 in your config.
        #
        # Output:
        # (depth*4, 4, 4)
        # = (64,4,4)
        # = 1024 features.

        self.network = nn.Sequential(

            nn.Conv2d(
                observation_shape[0],
                self.config.depth,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.Conv2d(
                self.config.depth,
                self.config.depth * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.Conv2d(
                self.config.depth * 2,
                self.config.depth * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.Conv2d(
                self.config.depth * 4,
                self.config.depth * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,
        )

        self.network.apply(initialize_weights)

    def forward(self, x):
        return horizontal_forward(
            self.network,
            x,
            input_shape=self.observation_shape,
        )

class Decoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()

        self.config = config.parameters.dreamer.decoder

        self.stochastic_size = config.parameters.dreamer.stochastic_size
        self.deterministic_size = config.parameters.dreamer.deterministic_size

        activation = getattr(nn, self.config.activation)()

        self.observation_shape = observation_shape

        self.network = nn.Sequential(

            nn.Linear(
                self.deterministic_size + self.stochastic_size,
                self.config.depth * 4 * 4 * 4,
            ),

            nn.Unflatten(
                1,
                (self.config.depth * 4, 4, 4),
            ),

            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                self.config.depth * 4,
                self.config.depth * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                self.config.depth * 2,
                self.config.depth,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            activation,

            nn.ConvTranspose2d(
                self.config.depth,
                observation_shape[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )

        self.network.apply(initialize_weights)

    def forward(self, posterior, deterministic):

        x = horizontal_forward(
            self.network,
            posterior,
            deterministic,
            output_shape=self.observation_shape,
        )

        return create_normal_dist(
            x,
            std=1,
            event_shape=len(self.observation_shape),
        )