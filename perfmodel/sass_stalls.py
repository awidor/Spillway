"""Where a kernel's warps wait, from Nsight Compute's per-instruction stall samples.

    ncu -o rep --kernel-name regex:NAME --launch-count 1 --section SourceCounters --section WarpStateStats ./app
    ncu -i rep.ncu-rep --page source --csv --print-source sass > sass.csv
    python3 perfmodel/sass_stalls.py sass.csv [top]

Prints the samples per opcode, per stall reason, and the hottest instructions with their main stall reasons.
"""
import csv
import sys
from collections import Counter

rows = list(csv.reader(open(sys.argv[1], encoding='utf-8', errors='replace')))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
hdr = next(i for i, r in enumerate(rows) if r and r[0] == 'Address')
cols = rows[hdr]
data = [r for r in rows[hdr + 1:] if len(r) == len(cols) and r[0].startswith('0x')]
ci = {c: i for i, c in enumerate(cols)}
stall_cols = [c for c in cols if c.startswith('stall_') and not c.endswith('(Not Issued)')]
total = sum(int(r[ci['Warp Stall Sampling (All Samples)']] or 0) for r in data)
by_op, by_reason = Counter(), Counter()
for r in data:
    n = int(r[ci['Warp Stall Sampling (All Samples)']] or 0)
    op = r[ci['Source']].split()[0].split('.')[0] if r[ci['Source']].split() else '?'
    if op.startswith('@'):
        op = r[ci['Source']].split()[1].split('.')[0]
    by_op[op] += n
    for c in stall_cols:
        by_reason[c[6:]] += int(r[ci[c]] or 0)
print(f'{total} samples')
print('by opcode: ' + ', '.join(f'{k} {100 * v / total:.1f}%' for k, v in by_op.most_common(16)))
print('by reason: ' + ', '.join(f'{k} {100 * v / total:.1f}%' for k, v in by_reason.most_common(10)))
wf = [(int(r[ci['L1 Wavefronts Shared']] or 0), int(r[ci['L1 Wavefronts Shared Excessive']] or 0), r) for r in data]
wf_total, wf_excess = sum(w for w, _, _ in wf), sum(e for _, e, _ in wf)
print(f'shared-memory wavefronts: {wf_total}, of them excess (bank conflicts) {wf_excess} ({100 * wf_excess / max(wf_total, 1):.1f}%)')
for w, e, r in sorted(wf, key=lambda x: -x[0])[:12]:
    print(f'  {100 * w / max(wf_total, 1):5.1f}% of wavefronts, excess {e:>10}  {r[ci["Address"]][-5:]}  {r[ci["Source"]][:60]}')
print('hottest instructions:')
for r in sorted(data, key=lambda r: -int(r[ci['Warp Stall Sampling (All Samples)']] or 0))[:top]:
    n = int(r[ci['Warp Stall Sampling (All Samples)']] or 0)
    reasons = sorted(((int(r[ci[c]] or 0), c[6:]) for c in stall_cols), reverse=True)[:2]
    print(f'  {100 * n / total:5.1f}%  {r[ci["Address"]][-5:]}  {r[ci["Source"]][:60]:60s}  '
          + ', '.join(f'{name} {v}' for v, name in reasons if v))
