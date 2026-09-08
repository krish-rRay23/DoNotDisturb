import re

with open('src/adaptive_plasticity/m5.py', 'r') as f:
    content = f.read()

# Use regex to find the exact section
pattern = re.compile(r'(# --- M10 adaptive plasticity controller step ---.*?)(# --- evaluation ---)', re.DOTALL)
match = re.search(r'(# --- M10 adaptive plasticity controller step ---.*?)(# --- evaluation ---)', content, re.DOTALL)

if match:
    old = match.group(1)
    print('Found old text, length:', len(match.group(1)))
    
    new_text = """# --- M10 adaptive plasticity controller step ---
                if adaptive_controller is not None:
                    # Gather M6-style diagnostics from current agent state.
                    # These are online-available; no future information used.
                    # repr_change: cosine similarity of policy parameters (temporal)
                    # perf_change: return delta from evaluation
                    # activation_dormancy: fraction of near-zero policy activations
                    # param_magnitudes: L2 norms of trainable parameters

                    # Compute diagnostics
                    try:
                        dorm_val = float(activation_dormancy(agent.policy, 0.001))
                    except Exception:
                        dorm_val = 0.0

                    # Compute temporal representation change
                    repr_val = 0.0
                    try:
                        repr_val = float(adaptive_controller._compute_temporal_repr_change(agent))
                    except Exception:
                        repr_val = 0.0

                    # Get current normalized performance for perf_change calculation
                    current_norm = metrics["normalized"][-1] if metrics["normalized"] else 0.0

                    # Update the controller and get intervention strength + perf_change
                    # using the corrected M10 rule: 0.4*repr - 0.4*perf - 0.2*dormancy
                    strength = adaptive_controller.update(
                        diagnostics={},
                        agent=agent,
                        current_perf=current_norm,
                    )

                    # Modulate the fixed intervention's severity for subsequent steps.
                    # The AdaptivePlasticityController returns strength in
                    # [min_severity, max_severity]; clip to [0,1] range.
                    if intervention is not None and isinstance(intervention, FixedIntervention):
                        intervention.severity = float(np.clip(strength, 0.0, 1.0))

                    # Get the actual perf_change delta used by the controller
                    perf_change = 0.0
                    if adaptive_controller._history:
                        perf_change = adaptive_controller._history[-1].get("perf_change", 0.0)

                    # Log diagnostics (single record per controller update)
                    if diag_log_file is None:
                        import json
                        diag_log_path = Path("results") / f"diagnostics_{out_name}.jsonl"
                        diag_log_file = open(diag_log_path, "w", encoding="utf-8")

                    # Log diagnostic record (single record per controller update)
                    diag_record = {
                        "step": online_step,
                        "shift_active": intervention.is_shifted(online_step) if intervention else False,
                        "controller_type": "adaptive" if adaptive_controller is not None else "fixed",
                        "repr_change": repr_val,
                        "perf_change": adaptive_controller._history[-1].get("perf_change", 0.0) if adaptive_controller._history else 0.0,
                        "activation_dormancy": dorm_val,
                        "param_magnitudes": param_magnitudes(agent),
                        "controller_strength": strength,
                        "evaluation_return": metrics["returns"][-1] if metrics["returns"] else None,
                        "evaluation_normalized": metrics["normalized"][-1] if metrics["normalized"] else None,
                    }
                    if diag_log_file is None:
                        import json
                        diag_log_path = Path("results") / f"diagnostics_{out_name}.jsonl"
                        diag_log_file = open(diag_log_path, "w", encoding="utf-8")

                    diag_log_file.write(json.dumps(diag_record) + "\n")
                    diag_log_file.flush()

                    # Update reference agent for next step
                    adaptive_controller.set_reference_agent(agent)

        # --- evaluation ---
"""

    if old in content:
        new_content = content.replace(old, new_text)
        with open('src/adaptive_plasticity/m5.py', 'w') as f:
            f.write(new_content)
        print("Successfully replaced the section")
    else:
        print("Old text not found exactly")
        # Find the difference
        idx = content.find("# --- M10 adaptive plasticity controller step ---")
        if idx >= 0:
            print("Found start at", idx)
            print("Content around start:", content[idx:idx+200])
        else:
            print("Start not found")
        idx = content.find("# --- evaluation ---")
        if idx >= 0:
            print("Found eval at", idx)
            print("Content around:", content[idx-100:idx+100])