"""
Task-specific goal states for the dm_control suite.

The suite tasks are *not* natively goal-conditioned, so LeWorldModel (see
``lewm.py``) needs a way to turn a task into a single goal observation to plan
towards. This module builds that goal by driving the underlying MuJoCo
``physics`` into a hand-specified goal configuration and asking the task's own
``get_observation`` for the resulting observation - so the goal vector has
exactly the same layout (same keys, same order, same flattening) as the
observations the agent sees at run time (see ``env.DMCEnv._obs``).

``goal_observation(env)`` is the entry point. It

  1. saves the live physics state,
  2. sets qpos / qvel to the goal pose (per-domain handler below),
  3. runs ``physics.forward()`` (kinematics only - no integration) and reads
     ``task.get_observation(physics)``,
  4. restores the live physics state so the episode is undisturbed.

Only a curated set of domains is supported; the goal poses for the locomotion
domains (``walker``, ``hopper``) are approximate "stand still upright" targets
and are easy to tune here. Unsupported tasks raise ``NotImplementedError`` with
the list of what *is* supported.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Reaching the underlying dm_control objects through the wrapper stack.
# ---------------------------------------------------------------------------

def _dmc_handles(env):
    """Return ``(physics, task)`` for a dm_control-backed ``env`` (env.DMCEnv).

    dm_control's ``base.Wrapper`` (and this repo's wrappers) forward attribute
    access to the wrapped env, so ``env.env.physics`` / ``env.env.task`` resolve
    straight through ``action_scale`` -> ``ActionRepeat`` -> ``ActionDType`` ->
    ``control.Environment``.
    """
    inner = getattr(env, "env", env)
    # Be defensive about the exact wrapper chain.
    seen = set()
    while not (hasattr(inner, "physics") and hasattr(inner, "task")) and id(inner) not in seen:
        seen.add(id(inner))
        nxt = getattr(inner, "_env", None) or getattr(inner, "env", None)
        if nxt is None:
            break
        inner = nxt
    if not (hasattr(inner, "physics") and hasattr(inner, "task")):
        raise TypeError("could not find a dm_control physics/task under this env")
    return inner.physics, inner.task


def _domain_name(task):
    """dm_control domain string, e.g. 'cartpole', 'ball_in_cup', 'point_mass'."""
    return type(task).__module__.rsplit(".", 1)[-1]


def _flatten_obs(obs_dict):
    """Match env.DMCEnv._obs: concat of the observation values, flattened."""
    return np.concatenate([np.asarray(v, np.float32).ravel() for v in obs_dict.values()])


# ---------------------------------------------------------------------------
# Per-domain goal poses. Each handler mutates ``physics.data`` in place; it may
# assume qpos/qvel have already been zeroed.
# ---------------------------------------------------------------------------

def _goal_cartpole(physics, task):
    # Cart centred, pole(s) upright, at rest. Zeroed qpos already gives this
    # (hinge angle 0 == upright, slider 0 == centre).
    pass


def _goal_pendulum(physics, task):
    # Hinge angle 0 == balanced upright; reset pose (angle pi) hangs down.
    pass


def _goal_acrobot(physics, task):
    # Both links pointing straight up.
    physics.named.data.qpos["shoulder"] = np.pi
    physics.named.data.qpos["elbow"] = 0.0


def _goal_ball_in_cup(physics, task):
    nd = physics.named.data
    nd.qpos["ball_x"] = nd.qpos["cup_x"]
    nd.qpos["ball_z"] = nd.qpos["cup_z"] - 0.05   # nestled just inside the cup


def _goal_point_mass(physics, task):
    # The two slide joints are the mass's (x, y); the target geom position is
    # fixed by initialize_episode for this episode.
    tgt = physics.named.model.geom_pos["target"][:2]
    physics.named.data.qpos["root_x"] = tgt[0]
    physics.named.data.qpos["root_y"] = tgt[1]


def _goal_reacher(physics, task):
    # Put the finger on the target. 2-DoF arm -> brute-force the joint grid
    # (cheap, and dodges IK sign/offset ambiguities in the model frame).
    tgt = physics.named.data.geom_xpos["target"][:2].copy()
    grid = np.linspace(-np.pi, np.pi, 121)
    best, best_q = np.inf, (0.0, 0.0)
    for a in grid:
        physics.named.data.qpos["shoulder"] = a
        for b in grid:
            physics.named.data.qpos["wrist"] = b
            physics.forward()
            d = np.linalg.norm(physics.named.data.geom_xpos["finger"][:2] - tgt)
            if d < best:
                best, best_q = d, (a, b)
    physics.named.data.qpos["shoulder"] = best_q[0]
    physics.named.data.qpos["wrist"] = best_q[1]


def _goal_finger(physics, task):
    # Turn tasks only: rotate the spinner so its tip sits on the target site.
    try:
        tgt = physics.named.data.site_xpos["target"].copy()
    except KeyError:
        raise NotImplementedError("finger goal is only defined for the turn tasks")
    best, best_a = np.inf, 0.0
    for a in np.linspace(-np.pi, np.pi, 361):
        physics.named.data.qpos["hinge"] = a
        physics.forward()
        d = np.linalg.norm(physics.named.data.site_xpos["tip"] - tgt)
        if d < best:
            best, best_a = d, a
    physics.named.data.qpos["hinge"] = best_a


def _goal_walker(physics, task):
    # Stand still, torso upright at standing height (walker._STAND_HEIGHT ~ 1.2).
    physics.named.data.qpos["rootz"] = 1.25


def _goal_hopper(physics, task):
    # Stand still, upright (hopper._STAND_HEIGHT ~ 0.6). Approximate.
    physics.named.data.qpos["rootz"] = 0.95


_REGISTRY = {
    "cartpole": _goal_cartpole,
    "pendulum": _goal_pendulum,
    "acrobot": _goal_acrobot,
    "ball_in_cup": _goal_ball_in_cup,
    "point_mass": _goal_point_mass,
    "reacher": _goal_reacher,
    "finger": _goal_finger,
    "walker": _goal_walker,
    "hopper": _goal_hopper,
}


def supported_domains():
    return sorted(_REGISTRY)


def goal_observation(env):
    """Goal observation vector for a dm_control-backed env, in run-time layout.

    Must be called *after* ``env.reset()`` (some goals - reacher, finger,
    point_mass - depend on the per-episode target). Does not disturb the live
    episode.
    """
    physics, task = _dmc_handles(env)
    domain = _domain_name(task)
    handler = _REGISTRY.get(domain)
    if handler is None:
        raise NotImplementedError(
            f"no goal state defined for dm_control domain {domain!r}; "
            f"supported: {supported_domains()}"
        )

    # Snapshot the live sim state so the running episode is left untouched.
    saved_qpos = physics.data.qpos.copy()
    saved_qvel = physics.data.qvel.copy()
    saved_act = physics.data.act.copy() if int(physics.model.na) else None
    try:
        physics.data.qpos[:] = 0.0
        physics.data.qvel[:] = 0.0
        handler(physics, task)
        physics.data.qvel[:] = 0.0          # goal is always "at rest"
        physics.forward()
        goal = _flatten_obs(task.get_observation(physics))
    finally:
        physics.data.qpos[:] = saved_qpos
        physics.data.qvel[:] = saved_qvel
        if saved_act is not None:
            physics.data.act[:] = saved_act
        physics.forward()
    return goal.astype(np.float32)
