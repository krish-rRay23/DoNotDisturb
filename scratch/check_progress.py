import glob
import json
import os
import time

summaries = glob.glob('results/final_study/*/summary.json')
now = time.time()
print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
print(f"Total completed runs: {len(summaries)} / 150\n")

print("Completed runs ordered by completion time:")
for s in sorted(summaries, key=os.path.getmtime):
    mtime_val = os.path.getmtime(s)
    mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime_val))
    mins_ago = (now - mtime_val) / 60.0
    with open(s) as fp:
        d = json.load(fp)
    print(f"  {mtime_str} ({mins_ago:4.1f}m ago) | {d.get('output_name')} | runtime: {d.get('runtime_seconds'):.1f}s ({d.get('runtime_seconds')/60:.2f}m)")

# Check directories in final_study without summary.json
all_dirs = [d for d in glob.glob('results/final_study/*') if os.path.isdir(d)]
active_dirs = [d for d in all_dirs if not os.path.exists(os.path.join(d, 'summary.json'))]
print(f"\nCurrently active / in-progress directories: {len(active_dirs)}")
for ad in active_dirs:
    print(f"  [IN PROGRESS] {os.path.basename(ad)}")
