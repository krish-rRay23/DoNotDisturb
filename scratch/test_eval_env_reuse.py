"""Test Gym MuJoCo eval_env reuse vs fresh creation difference."""
import gymnasium as gym
from adaptive_plasticity.m5 import make_env
from adaptive_plasticity.final_runner import run_final_cell, _evaluate, IQLAgent

def test_reuse_vs_fresh():
    # Fresh agent
    agent = IQLAgent(17, 6)

    # Env 1: reused env (eval step 50 then eval step 100)
    env1, _ = make_env("halfcheetah-medium-v2", seed=42 + 10000)
    ret1_step50 = _evaluate(agent, env1, seed=42 + 0, episodes=10, device="cpu")
    ret1_step100 = _evaluate(agent, env1, seed=42 + 50, episodes=10, device="cpu")

    # Env 2: fresh env (only eval step 100)
    env2, _ = make_env("halfcheetah-medium-v2", seed=42 + 10000)
    ret2_step100 = _evaluate(agent, env2, seed=42 + 50, episodes=10, device="cpu")

    print(f"Reused env step 100 ret: {ret1_step100}")
    print(f"Fresh env step 100 ret:  {ret2_step100}")
    print(f"Diff: {abs(ret1_step100 - ret2_step100)}")

if __name__ == "__main__":
    test_reuse_vs_fresh()
