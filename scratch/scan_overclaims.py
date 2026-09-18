import re

with open('research_paper/iclr2027/iclr2027_conference.tex', encoding='utf-8') as f:
    tex = f.read()

terms = [
    'if and only if',
    'essential safeguard',
    'necessary circuit breaker',
    'circuit breaker',
    'proves',
    'first systematic investigation',
    'provably',
    'safeguard'
]

for t in terms:
    matches = list(re.finditer(re.escape(t), tex, re.IGNORECASE))
    print(f"Term '{t}': {len(matches)} occurrences")
    for m in matches:
        start = max(0, m.start() - 50)
        end = min(len(tex), m.end() + 50)
        snippet = tex[start:end].replace('\n', ' ')
        print(f"   Line ~{tex[:m.start()].count(chr(10))+1}: ...{snippet}...")
