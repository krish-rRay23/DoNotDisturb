import glob
import json
import os

brain_dir = os.path.expanduser(r'~\.gemini\antigravity-ide\brain')
logs = glob.glob(os.path.join(brain_dir, '*', '.system_generated', 'logs', '*.jsonl'))

for log in logs:
    with open(log, encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            low = line.lower()
            if any(w in low for w in ["ma'am", "mam", "supervisor", "advisor", "professor"]):
                if '"type":"user_input"' in low or '"type": "user_input"' in low:
                    try:
                        d = json.loads(line)
                        content = d.get('content', '')
                        if any(w in content.lower() for w in ["ma'am", "mam", "supervisor", "advisor", "professor"]):
                            print(f"{log} [Line {i}]:")
                            print(content)
                            print("="*60)
                    except Exception:
                        pass
