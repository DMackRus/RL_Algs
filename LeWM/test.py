import os

# Headless GPU rendering (GLFW fails inside the container). Must be set before
# dm_control is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import yaml
import matplotlib.pyplot as plt
import numpy as np
from env import make_env
from goals import goal_observation, render_goal


def main(num_episodes=3):
    # Load the goal observation for an example task and then show what the goal observation looks like
    test_folder = "configs/default"
    yaml_name = "config.yaml"

    config_filepath = test_folder + "/" + yaml_name
    with open(config_filepath, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    env = make_env(config)

    # Goals can depend on the per-episode target (reacher, finger, point_mass),
    # so check a few resets: start state on the left, goal pose on the right.
    fig, axes = plt.subplots(num_episodes, 2, figsize=(8, 4 * num_episodes), squeeze=False)
    np.set_printoptions(precision=3, suppress=True)

    for i in range(num_episodes):
        obs = env.reset()
        goal = goal_observation(env)
        start_img = env.render(mode="rgb_array")
        goal_img = render_goal(env)

        dist = np.linalg.norm(np.asarray(obs, np.float32) - goal)
        print(f"Episode {i}")
        print(f"  start obs: {np.asarray(obs)}")
        print(f"  goal obs:  {goal}")
        print(f"  |obs - goal| = {dist:.3f}  (success threshold {config.get('goal_success_dist', 0.1)})")

        axes[i, 0].imshow(start_img)
        axes[i, 0].set_title(f"Episode {i}: start state")
        axes[i, 1].imshow(goal_img)
        axes[i, 1].set_title(f"Episode {i}: goal pose (|obs - goal| = {dist:.2f})")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(f"Goal check: {config['task']}")
    fig.tight_layout()
    out_path = os.path.join(test_folder, "goal_check.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
