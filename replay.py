import pickle
import matplotlib.pyplot as plt

with open("data/episodes.pkl", "rb") as f:
    episodes = pickle.load(f)

print(f"Loaded {len(episodes)} episodes")


print("Number of episodes:", len(episodes))

print("Length of first episode:", len(episodes[0]))

print("Keys:", episodes[0][0].keys())

episode = episodes[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].imshow(episode[50]["obs"])
ax[0].set_title("Frame t")
ax[0].axis("off")

ax[1].imshow(episode[51]["obs"])
ax[1].set_title("Frame t+1")
ax[1].axis("off")

plt.tight_layout()
plt.savefig("frame_comparison.png")