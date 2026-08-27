from typing import List
import torch
import torch.nn as nn

# -------------------------
# Encoder
# -------------------------
class Encoder(nn.Module):
    """
    Input:  (B, 3, 64, 64)
    Output: (B, embedding_dim) + skips
    """

    def __init__(self, in_channels: int = 3, embedding_dim: int = 100):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 64 -> 32
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.ReLU(inplace=True),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)   # (B,32,32,32)
        x = self.conv2(x)  # (B,64,16,16)
        x = self.conv3(x)  # (B,128,8,8)

        z = self.flatten(x)
        z = self.fc(z)

        return z


# -------------------------
# Decoder
# -------------------------
class Decoder(nn.Module):
    """
    Input:  (B, embedding_dim)
    Output: (B, 3, 64, 64)
    """

    def __init__(self, embedding_dim: int = 100, out_channels: int = 3):
        super().__init__()

        self.fc = nn.Linear(embedding_dim, 128 * 8 * 8)

        # 8 -> 16
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                64,
                4, 2, 1
            ),
            nn.ReLU(inplace=True),
        )

        # 16 -> 32
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(
                64,
                32,
                4, 2, 1
            ),
            nn.ReLU(inplace=True),
        )

        # 32 -> 64
        self.up3 = nn.ConvTranspose2d(
            32,
            out_channels,
            4, 2, 1
        )

    def forward(self, x):

        x = self.fc(x)  # (B, 128*8*8)
        x = x.view(-1, 128, 8, 8)

        # 8 -> 16
        x = self.up1(x)

        # 16 -> 32
        x = self.up2(x)

        # 32 -> 64
        x = self.up3(x)

        return x


# -------------------------
# Test
# -------------------------
if __name__ == "__main__":

    encoder = Encoder(in_channels=3, embedding_dim=100)
    decoder = Decoder(embedding_dim=100, out_channels=3)

    x = torch.randn(4, 3, 64, 64)
    z = encoder(x)
    out = decoder(z)

    print(out.shape)  # (4, 3, 64, 64)