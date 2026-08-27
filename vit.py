import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbedding(nn.Module):
    """
    Splits image into patches and embeds them.
    Input:  (B, C, H, W)
    Output: (B, N, D)
    """

    def __init__(self, in_channels=3, patch_size=8, embed_dim=256):
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2)  # (B, D, N)
        x = x.transpose(1, 2)  # (B, N, D)
        return x


# class TransformerBlock(nn.Module):
#     def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
#         super().__init__()

#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

#         self.norm2 = nn.LayerNorm(embed_dim)

#         hidden_dim = int(embed_dim * mlp_ratio)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#         )

#     def forward(self, x):
#         # Attention block
#         x_norm = self.norm1(x)
#         attn_out, _ = self.attn(x_norm, x_norm, x_norm)
#         x = x + attn_out

#         # MLP block
#         x = x + self.mlp(self.norm2(x))

#         return x


class ViTEncoder(nn.Module):
    """
    ViT encoder for 2x64x64 inputs.
    Outputs a single latent vector.
    """

    def __init__(
        self,
        in_channels=3,
        image_size=64,
        patch_size=8,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        latent_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # projection to latent space
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # x: (B, 2, 64, 64)

        B = x.shape[0]

        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        x = x + self.pos_embed
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_out = x[:, 0]  # CLS token
        latent = self.to_latent(cls_out)

        return latent

class PatchDecoder(nn.Module):
    """
    Turns tokens back into image patches.
    """

    def __init__(self, embed_dim=256, patch_size=8, out_channels=2):
        super().__init__()

        self.patch_size = patch_size
        self.out_channels = out_channels

        self.proj = nn.Linear(embed_dim, patch_size * patch_size * out_channels)

    def forward(self, x, num_patches_h, num_patches_w):
        """
        x: (B, N, D)
        returns: (B, C, H, W)
        """
        B, N, D = x.shape

        x = self.proj(x)  # (B, N, P*P*C)

        x = x.view(
            B,
            num_patches_h,
            num_patches_w,
            self.out_channels,
            self.patch_size,
            self.patch_size
        )

        # rearrange to image
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(
            B,
            self.out_channels,
            num_patches_h * self.patch_size,
            num_patches_w * self.patch_size
        )

        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(embed_dim)

        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDecoder(nn.Module):
    """
    ViT-style decoder for 2x64x64 images.
    """

    def __init__(
        self,
        image_size=64,
        patch_size=8,
        embed_dim=256,
        depth=4,
        num_heads=8,
        latent_dim=256,
        out_channels=3,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        # latent → token
        self.latent_to_token = nn.Linear(latent_dim, embed_dim)

        # learnable "seed" tokens (like CLS expanded)
        self.query_tokens = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self.to_img = PatchDecoder(embed_dim, patch_size, out_channels)

    def forward(self, z):
        """
        z: (B, latent_dim)
        """

        B = z.shape[0]

        # project latent
        z = self.latent_to_token(z)  # (B, D)

        # expand latent into patch tokens
        x = self.query_tokens.expand(B, -1, -1)  # (B, N, D)

        # inject global info
        x = x + z.unsqueeze(1)

        # transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # reshape back to image
        img = self.to_img(x, self.grid_size, self.grid_size)

        return img