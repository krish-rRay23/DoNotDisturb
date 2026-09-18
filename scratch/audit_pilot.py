import json
import torch
from pathlib import Path

pilot_dir = Path("results/pilot")
manifest = [json.loads(line) for line in open(pilot_dir / "PILOT_manifest.jsonl")]

print(f"PILOT RESULTS AUDIT ({len(manifest)} cells):")
print(f"{'Arm':<15} | {'Shift':<14} | {'Applied':<8} | {'Score':<8} | {'Returns (5 evals)'}")
print("-" * 80)

for m in manifest:
    name = m["output_name"]
    ctrl = m["controller"]
    shift = m["shift"]
    summary = json.loads((pilot_dir / name / "summary.json").read_text())
    applied = summary.get("applied_intervention_steps", -1)
    norm = summary.get("normalized", [])
    ret = summary.get("returns", [])
    score = m.get("score", norm[-1] if norm else 0.0)
    ret_str = ", ".join(f"{r:.1f}" for r in ret)
    print(f"{ctrl:<15} | {shift:<14} | {applied:<8} | {score:<8.2f} | [{ret_str}]")

print("\n=== CRITIC WEIGHT DIVERGENCE AUDIT (MaxDiff Q1 layer 0) ===")
for shift in ("none", "obs_noise", "reward_scale"):
    sev = 0.0 if shift == "none" else (0.1 if shift == "obs_noise" else 0.5)
    name_none = f"PILOT-hopper-medium-v2-none-{shift}-sev{sev:g}-seed0"
    name_fixed = f"PILOT-hopper-medium-v2-fixed-{shift}-sev{sev:g}-seed0"
    name_gate = f"PILOT-hopper-medium-v2-capacity_gate-{shift}-sev{sev:g}-seed0"

    ckpt_none = torch.load(pilot_dir / name_none / "checkpoint.pt", map_location="cpu", weights_only=False)
    ckpt_fixed = torch.load(pilot_dir / name_fixed / "checkpoint.pt", map_location="cpu", weights_only=False)
    ckpt_gate = torch.load(pilot_dir / name_gate / "checkpoint.pt", map_location="cpu", weights_only=False)

    w_none = ckpt_none["critic"]["q1.network.0.weight"]
    w_fixed = ckpt_fixed["critic"]["q1.network.0.weight"]
    w_gate = ckpt_gate["critic"]["q1.network.0.weight"]

    diff_fixed = (w_none - w_fixed).abs().max().item()
    diff_gate = (w_none - w_gate).abs().max().item()
    print(f"Shift: {shift:<12} | MaxDiff(none vs fixed) = {diff_fixed:.6f} | MaxDiff(none vs gate) = {diff_gate:.6f}")

print("\n=== POLICY WEIGHT DIVERGENCE AUDIT (MaxDiff Policy Backbone layer 0) ===")
for shift in ("none", "obs_noise", "reward_scale"):
    sev = 0.0 if shift == "none" else (0.1 if shift == "obs_noise" else 0.5)
    name_none = f"PILOT-hopper-medium-v2-none-{shift}-sev{sev:g}-seed0"
    name_fixed = f"PILOT-hopper-medium-v2-fixed-{shift}-sev{sev:g}-seed0"
    name_gate = f"PILOT-hopper-medium-v2-capacity_gate-{shift}-sev{sev:g}-seed0"

    ckpt_none = torch.load(pilot_dir / name_none / "checkpoint.pt", map_location="cpu", weights_only=False)
    ckpt_fixed = torch.load(pilot_dir / name_fixed / "checkpoint.pt", map_location="cpu", weights_only=False)
    ckpt_gate = torch.load(pilot_dir / name_gate / "checkpoint.pt", map_location="cpu", weights_only=False)

    p_none = ckpt_none["policy"]["backbone.network.0.weight"]
    p_fixed = ckpt_fixed["policy"]["backbone.network.0.weight"]
    p_gate = ckpt_gate["policy"]["backbone.network.0.weight"]

    diff_p_fixed = (p_none - p_fixed).abs().max().item()
    diff_p_gate = (p_none - p_gate).abs().max().item()
    print(f"Shift: {shift:<12} | Policy MaxDiff(none vs fixed) = {diff_p_fixed:.6f} | Policy MaxDiff(none vs gate) = {diff_p_gate:.6f}")
