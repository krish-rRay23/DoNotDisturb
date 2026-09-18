"""Test Gymnasium MuJoCo reset seed behavior after stepping."""
import gymnasium as gym
from adaptive_plasticity.m5 import make_env

def test_gym_reset():
    env1, _ = make_env("halfcheetah-medium-v2", seed=42)
    obs1, _ = env1.reset(seed=100)
    for _ in range(10):
        obs1, _, _, _, _ = env1.step(env1.action_space.sample())
    obs1_reset, _ = env1.reset(seed=200)

    env2, _ = make_env("halfcheetah-medium-v2", seed=42)
    # env2 was NOT stepped 10 times before seed 200 reset
    obs2_reset, _ = env2.reset(seed=200)

    diff = (obs1_reset - obs2_reset)
    print("Max diff in obs after reset:", abs(diff).max())
    assert (obs1_reset == obs2_reset).all(), "Gym reset with seed was NOT bit-identical!"
    print("PASS: Gym reset with seed is bit-identical!")

if __name__ == "__main__":
    test_gym_reset()
