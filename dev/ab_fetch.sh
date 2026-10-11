#!/usr/bin/env bash
# A/B of one env switch on the dev server with the laptop's flags (128K context, int8 K/V, --kv-grow, MTP drafts) and
# --pcie-frac 1: 6K prompt, 128 tokens, alternating runs; prints decode speed and a hash of the generated tokens.
# EXTRA: more engine flags for both.
#   bash dev/devbox.sh run 'bash dev/ab_fetch.sh "STRATA_FETCH_FORK=0" "STRATA_FETCH_FORK=8" [reps]'
set -u
A=$1; B=$2; R=${3:-2}; EXTRA=${EXTRA:-}; P=${P:-/strata/prompts/p6000.ids}
M=/strata/models/Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf
for i in $(seq "$R"); do
  for e in "$A" "$B"; do
    out=$(printf 'GEN 128 %s\nQUIT\n' "$(cat "$P")" |
      timeout 300 env $e STRATA_DECODE_TIMING=1 LD_LIBRARY_PATH=/src/out/spillway/lib /src/out/engine/strata --serve \
        --pack /strata/packs/qwen36-bs306 --native $M --expert-profile /src/engine/data/expert-profile-qwen36.bin \
        --expert-cache auto --prefill auto --spec 3 --spec-min-p 0.5 --mtp $M --mtp-draft-vocab /src/engine/data/draft_vocab.bin \
        --max-context 131072 --kv int8 --kv-grow --pcie-frac 1 $EXTRA 2>&1)
    echo "$e: $(grep '^T ' <<<"$out" | md5sum | cut -c1-8) $(grep -oE '[0-9.]+ ms/window|[0-9.]+ tok/s\), drafts accepted [0-9]+ of [0-9]+' <<<"$out" | tr '\n' ' ')"
  done
done
