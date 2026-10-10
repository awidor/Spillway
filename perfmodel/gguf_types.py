"""Expert and MTP-layer tensor types of a GGUF, grouped by tensor role: which kernels the engine needs.
    python gguf_types.py model.gguf"""
import collections
import sys

from gguf import GGUFReader

by = collections.defaultdict(lambda: collections.defaultdict(list))
for t in GGUFReader(sys.argv[1]).tensors:
    parts = t.name.split(".", 2)
    if parts[0] == "blk" and ("_exps" in t.name or parts[1] == "40"):
        by[parts[2] if parts[1] != "40" else "mtp." + parts[2]][t.tensor_type.name].append(int(parts[1]))
for role, types in sorted(by.items()):
    print(role, "  ".join("%s: %d layer(s)%s" % (ty, len(ls), "" if len(types) == 1 else " " + str(ls))
                          for ty, ls in types.items()))
