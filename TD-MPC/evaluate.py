import torch as T
import yaml
from env import make_env
from tdmpc import TDMPC
from replay_buffer import Episode



def main():
    print("Evaluation begins")

if __name__ == "__main__":

    config_filepath = "configs/default/default.yaml"

    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = make_env(config)

    IMAGE_OBSERVATIONS = config["image_observations"]
    if IMAGE_OBSERVATIONS:
        state_dim = (3 * config["frame_stack"], 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    config["action_dim"] = action_dim
    config["state_dim"] = state_dim
    config["episode_length"] = 500


    agent = TDMPC(config)
    model_path = "configs/default/model_checkpoints/checkpoint_step15000.pt"
    agent.load(model_path)

    num_eval_episodes = config["eval_episodes"]

    for episode in range(num_eval_episodes):

        obs = env.reset()
        episode = Episode(config, obs)
        step = 0
        while not episode.done:
            step += 1
            action = agent.plan(obs, eval_mode = True, step=step, t0=episode.first)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            episode += (obs, action, reward, done)

        print(f"Episode {episode} finished with total reward: {episode.cumulative_reward}")