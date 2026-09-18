"""Test if mujoco.mj_forward after set_state fixes physics trajectory divergence without importing torch."""
import numpy as np
import gymnasium as gym
from adaptive_plasticity.m5 import make_env

def test_forward():
    env1, _ = make_env("halfcheetah-medium-v2", seed=42)
    env1.reset(seed=42)

    # Step env1 50 times
    for _ in range(50):
        action = env1.action_space.sample()
        obs1, reward1, terminated, truncated, _ = env1.step(action)

    qpos = env1.unwrapped.data.qpos.copy()
    qvel = env1.unwrapped.data.qvel.copy()
    elapsed = getattr(env1, "_elapsed_steps", 0)

    # Now step env1 1 step further (step 51)
    test_action = env1.action_space.sample()
    obs1_step51, r1, t1, tr1, _ = env1.step(test_action)

    # Test env2 restored WITHOUT mj_forward
    env2, _ = make_env("halfcheetah-medium-v2", seed=42)
    env2.reset(seed=42 + 50)
    env2.unwrapped.set_state(qpos, qvel)
    env2._elapsed_steps = elapsed
    obs2_step51, r2, t2, tr2, _ = env2.step(test_action)

    diff_without = np.max(np.abs(obs1_step51 - obs2_step51))
    print("Max diff WITHOUT mj_forward:", diff_without)

    # Test env3 restored WITH mj_forward
    import mujoco
    env3, _ = make_env("halfcheetah-medium-v2", seed=42)
    env3.reset(seed=42 + 50)
    env3.unwrapped.set_state(qpos, qvel)
    mujoco.mj_forward(env3.unwrapped.model, env3.unwrapped.data)
    env3._elapsed_steps = elapsed
    obs3_step51, r3, t3, tr3, _ = env3.step(test_action)

    diff_with = np.max(np.abs(obs1_step51 - obs3_step51))
    print("Max diff WITH mj_forward:   ", diff_with)

if __name__ == "__main__":
    test_forward()
