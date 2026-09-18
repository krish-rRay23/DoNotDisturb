import tempfile
from pathlib import Path
import numpy as np
import torch
from adaptive_plasticity.final_runner import _evaluate
from adaptive_plasticity.iql import IQLAgent

agent1 = IQLAgent(17, 6, device="cpu")
agent2 = IQLAgent(17, 6, device="cpu")
agent2.load_state_dict(agent1.state_dict())

ret1 = _evaluate(agent1, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")

# Do some dummy env steps in main process
import gymnasium as gym
dummy_env = gym.make("HalfCheetah-v5")
dummy_env.reset(seed=123)
for _ in range(100):
    dummy_env.step(dummy_env.action_space.sample())
dummy_env.close()

ret2 = _evaluate(agent2, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")

print("ret1:", ret1)
print("ret2:", ret2)
print("diff:", abs(ret1 - ret2))
