import numpy as np
import gymnasium as gym

# Test 1: Uninterrupted
env1 = gym.make("HalfCheetah-v5")
env1.reset(seed=42)
for _ in range(50):
    a = env1.action_space.sample()
    obs1, r1, term1, trunc1, _ = env1.step(a)

qpos1 = env1.unwrapped.data.qpos.copy()
qvel1 = env1.unwrapped.data.qvel.copy()
elapsed1 = env1._elapsed_steps

# Step 51 in Uninterrupted
action51 = np.ones(env1.action_space.shape, dtype=np.float32)
next_obs1, r51_1, term51_1, trunc51_1, _ = env1.step(action51)

# Test 2: Resumed
env2 = gym.make("HalfCheetah-v5")
env2.reset(seed=42 + 50)
env2.unwrapped.set_state(qpos1, qvel1)
env2._elapsed_steps = elapsed1

next_obs2, r51_2, term51_2, trunc51_2, _ = env2.step(action51)

print("Obs diff:", np.max(np.abs(next_obs1 - next_obs2)))
print("Reward diff:", abs(r51_1 - r51_2))
