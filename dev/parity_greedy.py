"""Phase 1 parity: greedy continuations of the engine copy against llama.cpp on the same GGUF (RTX 3080 test PC, WSL).
Three prompts (code, chat, German), N tokens each; the engine with MTP off and on. Reports the first token where
each continuation leaves llama.cpp's (IQ3-level rounding moves near-ties, docs/QWEN36.md; a late split is normal).
    ~/venv/bin/python dev/parity_greedy.py [N]"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine", "tools"))
os.environ["STRATA_GGUF_PY"] = os.path.join(ROOT, "engine", "third_party", "llama.cpp", "gguf-py")
import strata_tokenizer as ST  # noqa: E402

HOME = os.path.expanduser("~")
MODEL = os.environ.get("MODEL", HOME + "/models/Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf")
PACK = os.environ.get("PACK", HOME + "/packs/qwen36-bs306")
ENGINE = os.environ.get("ENGINE", HOME + "/eng/strata")
LIB = os.path.join(ROOT, "out", "spillway", "lib")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
PROMPTS = ["Write a Python function that merges two sorted lists into one sorted list.",
           "What are three good ways to stay focused while working from home?",
           "Erkläre in zwei Sätzen, wie ein Kühlschrank funktioniert."]
TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def engine(ids, extra):
    env = dict(os.environ, LD_LIBRARY_PATH=LIB)
    out = subprocess.run([ENGINE, "--pack", PACK, "--native", MODEL, "--expert-profile",
                          os.path.join(ROOT, "engine", "data", "expert-profile-qwen36.bin"), "--expert-cache", "4000",
                          "--prefill", "auto", "--max-context", "4096", "--max-new", str(N),
                          "--tokens", ",".join(map(str, ids)), *extra],
                         capture_output=True, text=True, env=env, timeout=600)
    line = next((l for l in out.stdout.splitlines() if l.startswith("output")), None)
    if line is None:
        raise RuntimeError("engine: no output line\n" + out.stderr[-2000:])
    return [int(x) for x in line.split(":", 1)[1].split()]


def llama(text):
    env = dict(os.environ, LD_LIBRARY_PATH=LIB)
    out = subprocess.run([os.path.join(ROOT, "out", "tools", "llama-completion"), "-m", MODEL, "-p", text, "-n", str(N),
                          "--temp", "0", "-ngl", "99", "--n-cpu-moe", "15", "-c", "4096", "-no-cnv",
                          "--no-display-prompt", "-sp", "--no-warmup"],
                         capture_output=True, text=True, env=env, timeout=600)
    return out.stdout


def first_split(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


tok = ST.Tokenizer.from_gguf(MODEL)
mtp = ["--mtp", MODEL, "--mtp-draft-vocab", os.path.join(ROOT, "engine", "data", "draft_vocab.bin"), "--spec", "3",
       "--spec-min-p", "0.5"]
for p in PROMPTS:
    text = TEMPLATE.format(p)
    ids = tok.encode(text, parse_special=True)
    ref = tok.encode(llama(text), parse_special=True)[:N]
    for name, extra in (("no MTP", ["--spec", "2"]), ("MTP k=3", mtp)):
        got = engine(ids, extra)
        at = first_split(got, ref)
        print("%-62s %-8s %s" % (p[:60], name, "identical over %d tokens" % min(len(got), len(ref)) if at is None
                                  else "first split at token %d of %d: %r vs llama.cpp %r"
                                  % (at, N, tok.decode(got[at:at + 4]), tok.decode(ref[at:at + 4]))))
