import torch as T
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import repeat
import matplotlib.pyplot as plt
from torchvision.datasets import OxfordIIITPet
from random import random
from torchvision.transforms import Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

to_tensor = [Resize((144, 144)), ToTensor()]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img = t(img)
        return img, target

def show_images(images, num_samples=20, cols=4):
    """ Plot some samples from the dataset. """
    plt.figure(figsize=(15, 15))
    idx = int(len(dataset) / num_samples)
    print(images)

    for i, img in enumerate(images):
        if i % idx == 0:
            plt.subplot(int(num_samples/cols)+1, cols, int(i/idx) + 1)
            plt.axis('off')
            plt.imshow(to_pil_image(img[0]))

    plt.show()


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128):
        super().__init__()
        self.patch_size = patch_size

        # Output shape of projection layer: (B, n_patches, emb_size)
        self.projection = nn.Sequential(
            # break down the image into s1 x s2 patches and flatten them
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x):
        return self.projection(x)

class Attention(nn.Module):
    """ Multi-head self-attention layer. """
    def __init__(self, emb_dim, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.att = nn.MultiheadAttention(embed_dim=emb_dim,
                                                num_heads=n_heads,
                                                dropout=dropout,
                                                batch_first=True)   # Changes expected order of input to batch first
        
        self.q = nn.Linear(emb_dim, emb_dim)
        self.k = nn.Linear(emb_dim, emb_dim)
        self.v = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        attn_output, attn_output_weights = self.att(q, k, v)
        return attn_output

class PreNorm(nn.Module):
    """ Applies layer normalization before the function 'fn' is applied. """
    def __init__(self, emb_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(emb_dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    """ Feed forward layer. """
    def __init__(self, emb_dim, expansion_size = 4, dropout=0.):
        super().__init__()
        hidden_dim = emb_dim * expansion_size
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward(self, x):
        return self.feed_forward(x)

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class ViT(nn.Module):
    """ Vision Transformer. """
    def __init__(self, ch=3, img_size=144, patch_size=8, emb_dim=32,
                    n_layers=4, out_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        # Attributes
        self.channels = ch
        self.height = img_size
        self.width = img_size
        self.patch_size = patch_size
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        expansion_size = 4  # Feed forward expansion size

        # Components
        self.patch_embedding = PatchEmbedding(ch, patch_size, emb_dim)

        # Learnable params
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(T.randn(1, num_patches + 1, emb_dim)) # Patch embeddings plus 1 for the CLS token
        self.cls_token = nn.Parameter(T.rand(1, 1, emb_dim)) # Classification token attends all other pathces and joins information into one token

        # Transformer encoder
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            transformer_block = nn.Sequential(
                ResidualAdd(PreNorm(emb_dim, Attention(emb_dim, n_heads = heads, dropout = dropout))),
                ResidualAdd(PreNorm(emb_dim, FeedForward(emb_dim, expansion_size, dropout = dropout))))
            self.layers.append(transformer_block)

        # Classification head
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, out_dim))


    def forward(self, img):

        # Patch embedding vectors
        x = self.patch_embedding(img)
        b, n, _ = x.shape # (Batches, num_patches, pathc_size)

        # Add CLS token to inputs
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = T.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n+1)]

        # Transformer layers
        for i in range(self.n_layers):
            x = self.layers[i](x)

        # Output based on classification token
        return self.head(x[:, 0, :])



if __name__ == "__main__":
    train_dataset = OxfordIIITPet(root='.', split="trainval", download=False, transforms=Compose(to_tensor))
    test_dataset = OxfordIIITPet(root='.', split="test", download=False, transforms=Compose(to_tensor))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    model = ViT(ch=3, img_size=144, patch_size=8, emb_dim=32, n_layers=4, out_dim=37, dropout=0.1, heads=2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = T.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

    num_epochs = 100
    eval_every = 5  # Evaluate on the test set every N epochs

    def evaluate(model, loader):
        """ Returns mean loss and accuracy of the model over a dataloader. """
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with T.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                total_loss += criterion(logits, labels).item() * labels.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / total, correct / total

    # Training statistics
    train_losses, train_accs = [], []
    test_epochs, test_losses, test_accs = [], [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            logits = model(imgs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_losses.append(epoch_loss / total)
        train_accs.append(correct / total)
        msg = f"Epoch [{epoch+1}/{num_epochs}], Train loss: {train_losses[-1]:.4f}, Train acc: {train_accs[-1]:.3f}"

        if (epoch + 1) % eval_every == 0 or epoch == num_epochs - 1:
            test_loss, test_acc = evaluate(model, test_loader)
            test_epochs.append(epoch + 1)
            test_losses.append(test_loss)
            test_accs.append(test_acc)
            msg += f", Test loss: {test_loss:.4f}, Test acc: {test_acc:.3f}"

        print(msg)

    # Plot training statistics
    epochs = range(1, num_epochs + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, train_losses, label="Train")
    ax_loss.plot(test_epochs, test_losses, "o-", label="Test")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.legend()

    ax_acc.plot(epochs, train_accs, label="Train")
    ax_acc.plot(test_epochs, test_accs, "o-", label="Test")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig("training_stats.png")
    plt.show()