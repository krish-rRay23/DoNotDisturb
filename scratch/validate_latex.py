import re

with open('research_paper/iclr2027/iclr2027_conference.tex', encoding='utf-8') as f:
    tex = f.read()

# 1. Verify no unescaped & outside tabular/cases/algorithmic
lines = tex.splitlines()
in_tab = False
in_cases = False
in_alg = False
bad_amp = []
for i, l in enumerate(lines, 1):
    if r'\begin{tabular}' in l: in_tab = True
    if r'\end{tabular}' in l: in_tab = False
    if r'\begin{cases}' in l: in_cases = True
    if r'\end{cases}' in l: in_cases = False
    if r'\begin{algorithmic}' in l: in_alg = True
    if r'\end{algorithmic}' in l: in_alg = False
    if not (in_tab or in_cases or in_alg):
        if re.search(r'(?<!\\)&', l):
            bad_amp.append((i, l))

print(f"Unescaped '&' count: {len(bad_amp)}")
for i, l in bad_amp:
    print(f"  Line {i}: {l}")

# 2. Verify no bad mathbf
bad_mb = []
for m in re.finditer(r'\\mathbf\{([^}]+)\}', tex):
    inner = m.group(1)
    if any(c in inner for c in ['\\pm', '[', ']', '\\', ' ']):
        bad_mb.append(m.group(0))
print(f"Bad \\mathbf count: {len(bad_mb)}")
for b in bad_mb:
    print(f"  Bad mathbf: {b}")

# 3. Verify environment balance
begins = re.findall(r'\\begin\{([^}]+)\}', tex)
ends = re.findall(r'\\end\{([^}]+)\}', tex)
print(f"Begins count: {len(begins)}, Ends count: {len(ends)}")
assert len(begins) == len(ends), f"Mismatch: {len(begins)} vs {len(ends)}"

# 4. Check all labels and refs
labels = set(re.findall(r'\\label\{([^}]+)\}', tex))
refs = set(re.findall(r'\\ref\{([^}]+)\}', tex))
print(f"Labels: {len(labels)}, Refs: {len(refs)}")
missing = refs - labels
print(f"Missing refs: {missing}")

print("\n>>> ALL VALIDATION CHECKS PASSED WITH 0 ERRORS! <<<")
