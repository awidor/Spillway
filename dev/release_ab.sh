#!/usr/bin/env bash
# Release comparison on the dev server: the previous kit's engine against this build, same flags as the laptop
# (128K context, int8 K/V, --kv-grow, MTP draft length 3), alternating runs.  Prints prompt read time, decode speed,
# drafts and a hash of the generated tokens.
#   bash dev/devbox.sh run 'bash dev/release_ab.sh /work/v030kit/bin/strata /src/out/engine/strata [reps] [extra flags]'
set -u
OLD=$1; NEW=$2; R=${3:-1}; shift 3 || shift $#
M=/strata/models/Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf
for P in p6000 p30000; do
  for i in $(seq "$R"); do
    for eng in "$OLD" "$NEW"; do
      out=$(printf 'GEN 128 %s\nQUIT\n' "$(cat /strata/prompts/$P.ids)" |
        timeout 600 env LD_LIBRARY_PATH=/src/out/spillway/lib "$eng" --serve \
          --pack /strata/packs/qwen36-bs306 --native $M --expert-profile /src/engine/data/expert-profile-qwen36.bin \
          --expert-cache auto --prefill auto --spec 3 --spec-min-p 0.5 --mtp $M --mtp-draft-vocab /src/engine/data/draft_vocab.bin \
          --max-context 131072 --kv int8 --kv-grow "$@" 2>&1)
      echo "$P $(basename "$(dirname "$(dirname "$eng")")"): $(grep '^T ' <<<"$out" | md5sum | cut -c1-8) $(grep -oE 'read in [0-9]+ ms|generated in [0-9]+ ms \([0-9.]+ tok/s\), drafts accepted [0-9]+ of [0-9]+' <<<"$out" | tr '\n' ' ')"
    done
  done
done
