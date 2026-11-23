# Documentation

`claim_log_checklist.md` maps each manuscript claim to the script and working-tree log that would carry it. Absent logs are `unverified`. The page does not recompute a score.

`claim_numeric_tokens.yaml` lists every decimal and `10^{n}` power printed in the manuscript, under one of those claim ids. `experiments/check_claim_coverage.py` fails when a new decimal is printed and is not in that map.

`run_guide.md` is the GPU sequence for the four confirm-gated matrices and for regenerating tables and figures from their outputs. It does not launch those runs.

`blockers.md` lists what is still blocked outside this repository: revised manuscript numbers, the GPU runs, Cal-QL/PEX/RO2O code, and an AntMaze adapter. Each section says what would unblock it. It does not add a score.

