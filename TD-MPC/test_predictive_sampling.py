import os

import torch as T
import gymnasium as gym
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

CHECKPOINT_PATH = "configs/default/model_checkpoints/checkpoint_round40.pt"
# CHECKPOINT_PATH = "lunar_lander_solved.pt"
# CHECKPOINT_PATH = "configs/testing_horizons/horizon_10/model_checkpoints/checkpoint_round195.pt"

DEVICE = T.device(
    "cuda" if T.cuda.is_available() else "cpu"
)

VIDEO_DIR = "configs/default/eval_videos"


def save_video(frames, path, fps=30):
    """Write a list of RGB uint8 (H, W, 3) frames to an mp4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"could not open video writer for {path}")
        return
    for f in frames:
        writer.write(cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"saved {path} ({len(frames)} frames @ {fps} fps)")


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

    # NOTE: train.py's collect_play_data() hard-codes MPPISampler and ignores
    # config["sampler"], so that's the planner the checkpoints were trained and
    # evaluated with. Match it here (set SAMPLER = "predictive-sampling" to
    # compare the weaker single-shot planner against the same world model).
    SAMPLER = "MPPI"
    Sampler = MPPISampler if SAMPLER == "MPPI" else PredictiveSampler

    def make_sampler():
        return Sampler(
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

    # Record frames whenever the env renders to rgb arrays (always the case in
    # image-observation mode). "human" render returns None -> nothing to save.
    record = render_mode == "rgb_array"
    if record:
        os.makedirs(VIDEO_DIR, exist_ok=True)
    video_fps = env.metadata.get("render_fps", 30)

    for episode in range(5):

        # Fresh sampler per episode -- matches collect_play_data() and resets the
        # warm-started nominal action sequence.
        trajectory_optimizer = make_sampler()

        state, info = env.reset()

        frames = []
        if record:
            frames.append(env.render())

        stack_push = None
        if IMAGE_OBSERVATIONS:
            # New frame stacker per episode. reset() copies the first frame
            # FRAME_STACK times so we start with a full (3*FRAME_STACK, 64, 64) stack.
            stack_reset, stack_push = make_frame_stacker(FRAME_STACK)
            state = stack_reset(process_image(frames[-1]))

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
            # print(f"Action: {action} | Predicted return: {predicted.item():8.2f}")

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            frame = env.render() if record else None
            if record:
                frames.append(frame)

            if IMAGE_OBSERVATIONS:
                state = stack_push(process_image(frame))


            # print(
            #     f"Predicted return: {predicted.item():8.2f} "
            #     f"| Reward: {reward:8.2f}"
            # )


        print(
            f"\nEpisode {episode+1} reward:",
            total_reward
        )

        if record:
            save_video(
                frames,
                os.path.join(VIDEO_DIR, f"episode_{episode + 1}.mp4"),
                fps=video_fps,
            )



    env.close()



if __name__ == "__main__":
    main()