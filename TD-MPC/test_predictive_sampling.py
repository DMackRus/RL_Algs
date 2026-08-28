import torch as T
import gymnasium as gym
import numpy as np

from models import (
    RepresentationModel,
    LatentDynamics,
    RewardPredictor,
    ValuePredictor,
    PolicyModel,
)

from planner import PredictiveSampler, MPPISampler
from utils import process_image, make_frame_stacker
import yaml

# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = "configs/default/model_checkpoints/checkpoint_round195.pt"
# CHECKPOINT_PATH = "lunar_lander_solved.pt"
# CHECKPOINT_PATH = "configs/testing_horizons/horizon_10/model_checkpoints/checkpoint_round195.pt"

DEVICE = T.device(
    "cuda" if T.cuda.is_available() else "cpu"
)

# ============================================================
# Main Evaluation
# ============================================================
def main():

    print("Using device:", DEVICE)

    print("Loading checkpoint...")

    checkpoint = T.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
    )

    config_filepath = "configs/default/default.yaml"
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    IMAGE_OBSERVATIONS = config["image_observations"]
    FRAME_STACK = config.get("frame_stack", 1)
    LATENT_DIM = config["latent_dim"]
    HIDDEN_DIM = config["hidden_dim"]

    # Image mode needs rgb frames from env.render(); "human" render returns None.
    render_mode = "rgb_array" if IMAGE_OBSERVATIONS else "human"
    # env = gym.make("LunarLanderContinuous-v3", render_mode=render_mode)
    env = gym.make("Walker2d-v5", render_mode=render_mode)
    # env = gym.make("BipedalWalker-v3", render_mode=render_mode)

    action_dim = env.action_space.shape[0]
    if IMAGE_OBSERVATIONS:
        state_dim = (3 * FRAME_STACK, 64, 64)  # (C, H, W)
    else:
        state_dim = env.observation_space.shape[0]

    # --------------------------------------------------------
    # Build models
    # --------------------------------------------------------

    representation_model = RepresentationModel(
        LATENT_DIM,
        state_dim,
        HIDDEN_DIM,
        image_state=IMAGE_OBSERVATIONS,
        frame_stack=FRAME_STACK,
    )


    latent_dynamics = LatentDynamics(
        LATENT_DIM,
        action_dim,
        hidden_dim=HIDDEN_DIM,
    )


    reward_predictor = RewardPredictor(
        LATENT_DIM,
        action_dim,
        hidden_dim=HIDDEN_DIM,
    )


    value_predictor = ValuePredictor(
        LATENT_DIM,
        action_dim,
        hidden_dim=HIDDEN_DIM,
    )


    policy_model = PolicyModel(
        LATENT_DIM,
        action_dim,
        hidden_dim=HIDDEN_DIM,
    )

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    representation_model.load_state_dict(
        checkpoint["representation_model"]
    )

    latent_dynamics.load_state_dict(
        checkpoint["latent_dynamics"]
    )

    reward_predictor.load_state_dict(
        checkpoint["reward_predictor"]
    )

    value_predictor.load_state_dict(
        checkpoint["value_predictor"]
    )

    policy_model.load_state_dict(
        checkpoint["policy_model"]
    )

    models = [
        representation_model,
        latent_dynamics,
        reward_predictor,
        value_predictor,
        policy_model,
    ]

    for model in models:
        model.to(DEVICE)
        model.eval()

    # Create the trajectory optimizer
    trajectory_optimizer = None
    if config["sampler"] == "MPPI":
        trajectory_optimizer = MPPISampler(
            config=config,
            representation_model=representation_model,
            dynamics_model=latent_dynamics,
            reward_model=reward_predictor,
            value_model=value_predictor,
            policy=policy_model,
            action_dim=action_dim,
            device=DEVICE,
        )
    elif config["sampler"] == "predictive-sampling":
        trajectory_optimizer = PredictiveSampler(
            config=config,
            representation_model=representation_model,
            dynamics_model=latent_dynamics,
            reward_model=reward_predictor,
            value_model=value_predictor,
            policy=policy_model,
            action_dim=action_dim,
            device=DEVICE,
        )

    # --------------------------------------------------------
    # Episodes
    # --------------------------------------------------------

    for episode in range(5):

        state, info = env.reset()

        stack_push = None
        if IMAGE_OBSERVATIONS:
            # New frame stacker per episode. reset() copies the first frame
            # FRAME_STACK times so we start with a full (3*FRAME_STACK, 64, 64) stack.
            stack_reset, stack_push = make_frame_stacker(FRAME_STACK)
            state = stack_reset(process_image(env.render()))

        done = False
        total_reward = 0

        while not done:

            # State -> latent

            if isinstance(state, np.ndarray):
                state_tensor = T.from_numpy(state).unsqueeze(0).float().to(DEVICE)
            else:
                state_tensor = state.unsqueeze(0).float().to(DEVICE)

            with T.no_grad():

                action, predicted = trajectory_optimizer.plan(
                    state_tensor
                )

            action = (action.cpu().numpy())

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if IMAGE_OBSERVATIONS:
                state = stack_push(process_image(env.render()))


            # print(
            #     f"Predicted return: {predicted.item():8.2f} "
            #     f"| Reward: {reward:8.2f}"
            # )


        print(
            f"\nEpisode {episode+1} reward:",
            total_reward
        )



    env.close()



if __name__ == "__main__":
    main()