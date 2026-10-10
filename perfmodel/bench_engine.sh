#!/usr/bin/env bash
# One decode benchmark of the engine copy on the proxy (WSL): a fixed chat prompt, N new tokens, greedy, through
# --serve so the engine prints its decode timing and GPU stage profile.
#   bash perfmodel/bench_engine.sh [N] [extra engine flags...]
# Env: ENGINE (default ~/eng/strata), PACK, MODEL, PROFILE=0 to drop STRATA_VERIFY_PROFILE (it slows decode a little).
set -eu
N=${1:-128}; shift || true
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENGINE=${ENGINE:-$HOME/eng/strata}
PACK=${PACK:-$HOME/packs/qwen36-bs306}
MODEL=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf}
PROMPT=${PROMPT:-$HOME/eng/prompt.ids}
if [ ! -s "$PROMPT" ]; then   # the chat template around one coding request, tokenized by the engine's tokenizer
  STRATA_GGUF_PY=$ROOT/engine/third_party/llama.cpp/gguf-py PYTHONPATH=$ROOT/engine/tools ~/venv/bin/python -c '
import sys, strata_tokenizer as ST
t = ST.Tokenizer.from_gguf(sys.argv[1])
msg = "Write a Python function that checks whether a number is prime, then explain how it works."
print(",".join(map(str, t.encode("<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                                  parse_special=True))))' "$MODEL" > "$PROMPT"
fi
prof=(); [ "${PROFILE:-1}" = 1 ] && prof=(STRATA_VERIFY_PROFILE=1)
printf 'GEN %s %s\nQUIT\n' "$N" "$(cat "$PROMPT")" |
  env "${prof[@]}" STRATA_DECODE_TIMING=1 LD_LIBRARY_PATH="$ROOT/out/s1kit/lib" \
  "$ENGINE" --serve --pack "$PACK" --native "$MODEL" --expert-profile "$ROOT/engine/data/expert-profile-qwen36.bin" \
    --expert-cache auto --prefill auto --spec 2 --max-context 4096 "$@" 2>&1 |
  grep -E "expert cache [0-9]+ slots|decode timing|GPU stages|tok/s|^DONE|^ERR|error|refus" || true
