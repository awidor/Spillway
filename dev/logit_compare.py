"""Compare next-token distributions dumped by STRATA_DUMP_FIRST_LOGITS (raw float32 logits, one row).

    python3 dev/logit_compare.py REF.bin OTHER.bin [OTHER.bin ...]

Prints, against REF: KL(ref || other) in nats, whether the top token agrees, the overlap of the top 5, the largest
probability change and the largest logit change.
"""
import sys

import numpy as np


def load(path):
    x = np.fromfile(path, dtype=np.float32).astype(np.float64)
    p = np.exp(x - x.max())
    return x, p / p.sum()


ref_x, ref_p = load(sys.argv[1])
ref_top = np.argsort(-ref_p)[:5]
print(f"{sys.argv[1]}: top token {ref_top[0]} p={ref_p[ref_top[0]]:.4f}")
for path in sys.argv[2:]:
    x, p = load(path)
    top = np.argsort(-p)[:5]
    kl = float(np.sum(ref_p * (np.log(ref_p + 1e-300) - np.log(p + 1e-300))))
    print(f"  vs {path}: KL {kl:.2e}, top-1 {'same' if top[0] == ref_top[0] else 'DIFFERENT'}, "
          f"top-5 overlap {len(set(top) & set(ref_top))}/5, max |dp| {np.abs(p - ref_p).max():.2e}, "
          f"max |dlogit| {np.abs(x - ref_x).max():.3f}")
