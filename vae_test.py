import os
import pickle

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vae import Encoder, Decoder


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FrameDataset(Dataset):
    """
    Dataset of individual frames extracted from collected episodes.
    """

    def __init__(self, path):

        with open(path, "rb") as f:
            episodes = pickle.load(f)

        self.frames = []

        for episode in episodes:
            for transition in episode:
                self.frames.append(transition["obs"])

        print(f"Loaded {len(self.frames)} frames.")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):

        frame = self.frames[idx]

        # Resize to 64x64
        frame = cv2.resize(frame, (64, 64))

        # Convert to float in [0,1]
        # frame = frame.astype("float32") / 255.0

        # Convert to float in [-1, 1]
        frame = (frame.astype("float32")  / 255.0 - 0.5) / 0.5

        # HWC -> CHW
        frame = torch.from_numpy(frame).permute(2, 0, 1)

        return frame


def save_reconstruction(encoder, decoder, dataset, device, epoch):
    """
    Saves an example reconstruction to disk.
    """

    encoder.eval()
    decoder.eval()

    with torch.no_grad():

        image = dataset[100].unsqueeze(0).to(device)

        z = encoder(image)
        reconstruction = decoder(z)

    original = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
    reconstruction = reconstruction.squeeze(0).cpu().permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(original)
    ax[0].set_title("Original")
    ax[0].axis("off")

    ax[1].imshow(reconstruction)
    ax[1].set_title(f"Epoch {epoch}")
    ax[1].axis("off")

    plt.tight_layout()

    os.makedirs("reconstructions", exist_ok=True)
    plt.savefig(f"reconstructions/epoch_{epoch:03d}.png")
    plt.close(fig)

    encoder.train()
    decoder.train()

def main():

    dataset = FrameDataset("data/episodes.pkl")

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        drop_last=True,
    )

    encoder = Encoder(embedding_dim=5).to(DEVICE)
    decoder = Decoder(embedding_dim=5).to(DEVICE)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=1e-3,
    )

    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    epochs = 500

    for epoch in range(epochs):

        encoder.train()
        decoder.train()

        running_loss = 0.0

        for images in loader:

            images = images.to(DEVICE)
            print(f"images shape: {images.shape}")

            z = encoder(images)
            reconstructions = decoder(z)
            

            loss = criterion(reconstructions, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(loader)

        print(
            f"Epoch {epoch+1:02d}/{epochs} "
            f"Loss: {avg_loss:.6f}"
        )

        if(epoch + 1) % 10 == 0:
            save_reconstruction(
                encoder,
                decoder,
                dataset,
                DEVICE,
                epoch + 1,
            )

    # Save weights
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
        },
        "autoencoder.pt",
    )

    print("Saved model to autoencoder.pt")

if __name__ == "__main__":
    main()