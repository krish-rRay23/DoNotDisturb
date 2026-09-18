"""Test evaluating the exact same agent twice with _evaluate."""
from adaptive_plasticity.final_runner import _evaluate, IQLAgent
from adaptive_plasticity.m5 import load_offline_data

def test_twice():
    data = load_offline_data("halfcheetah-medium-v2")
    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]

    agent = IQLAgent(obs_dim, act_dim, device="cpu")

    r1 = _evaluate(agent, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")
    r2 = _evaluate(agent, "halfcheetah-medium-v2", seed=92, episodes=10, device="cpu")

    print(f"Eval 1: {r1}")
    print(f"Eval 2: {r2}")
    print(f"Diff: {abs(r1 - r2)}")

if __name__ == "__main__":
    test_twice()
