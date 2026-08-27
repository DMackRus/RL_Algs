
import torch as T
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from models import RepresentationModel, LatentDynamics, RewardPredictor, ValuePredictor, PolicyModel


IMAGE_OBSERVATIONS = False  # Set to True if using image-based observations

def main():
    print("Loading model checkpoint...")

    env = gym.make("LunarLanderContinuous-v3", render_mode="human")

    # Load the model checkpoint
    checkpoint_path = "model_checkpoints/checkpoint_round25.pt"
    checkpoint = T.load(checkpoint_path, map_location=T.device("cpu"))

    latent_dim = 30
    hidden_dim = 256
    H = 5
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    representation_model = RepresentationModel(latent_dim, state_dim, hidden_dim, image_state=IMAGE_OBSERVATIONS)
    policy_model = PolicyModel(latent_dim, action_dim, hidden_dim)

    representation_model.load_state_dict(checkpoint["representation_model"])
    policy_model.load_state_dict(checkpoint["policy_model"])

    with T.no_grad():
        for i in range(5):  # Run 5 evaluation episodes
            state, info = env.reset()
            done = False
            total_reward = 0
            while not done:
                if IMAGE_OBSERVATIONS:
                    observation = process_image(env.render())
                    z = representation_model(observation.unsqueeze(0))
                else:
                    z = representation_model(T.from_numpy(state).unsqueeze(0).float())
                action = policy_model(z)
                action = action.squeeze(0).numpy()

                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = next_state

            print(f"Episode {i + 1}: Total Reward: {total_reward}")

if __name__ == "__main__":
    main()





