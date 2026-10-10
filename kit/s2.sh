#!/usr/bin/env bash
# Spillway S2 (PLAN.md section 5), supervised: install the engine on the box, pack the model, tune the few settings only
# the box can decide, start it as a service and check it through the API.
#   bash s2kit/s2.sh            install + tune + start          (asks for sudo once)
#   bash s2kit/s2.sh --undo     stop and remove the service and /opt/strata, restore the CPU governor
# The model is taken from this folder or ../s1kit/ (where S1 downloaded it), else downloaded here.
# The API: http://127.0.0.1:8095/v1 (OpenAI) on the box; HOST=0.0.0.0 bash s2.sh to listen on the network.
set -eu
KIT=$(cd "$(dirname "$0")" && pwd)
DEST=/opt/strata
USER_NAME=${SUDO_USER:-$(id -un)}
FILE=Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf
SIZE=13609247424
URL=https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/$FILE
HOST=${HOST:-127.0.0.1}
CTX=131072   # the target: at least 128K tokens of context
say() { echo "[$(date +%T)] $*"; }

if [ "${1:-}" = "--undo" ]; then
  sudo systemctl disable --now strata.service 2>/dev/null || true
  sudo rm -f /etc/systemd/system/strata.service && sudo systemctl daemon-reload
  if [ -f $DEST/undo.env ]; then
    . $DEST/undo.env
    for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -e "$g" ] && echo "$GOVERNOR" | sudo tee "$g" >/dev/null; done
    [ "$PERSISTENCE" = Disabled ] && sudo nvidia-smi -pm 0 >/dev/null || true
  fi
  sudo rm -rf $DEST
  say "undone: service removed, $DEST deleted, governor and persistence mode restored (the model file is kept)"
  exit 0
fi

# ---- checks
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${DRV%%.*}" -ge 525 ] 2>/dev/null || { echo "need NVIDIA driver 525+, have '$DRV'"; exit 1; }
command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "need Python 3.10+ (python3)"; exit 1; }
sudo true   # asks for the password once, here (sudo -v asks even with NOPASSWD)
python3 -c 'import venv, ensurepip' 2>/dev/null || { say "installing python3-venv"; timeout 300 sudo apt-get install -y python3-venv; }

# ---- the model: next to this kit, next to S1's, or downloaded here
M=
for c in "$KIT/$FILE" "$KIT/../s1kit/$FILE"; do [ "$(stat -c %s "$c" 2>/dev/null)" = $SIZE ] && { M=$(readlink -f "$c"); break; }; done
if [ -z "$M" ]; then M=$KIT/$FILE; say "downloading the model (13.6 GB)"; curl -fL -C - -o "$M" "$URL"; fi
say "model: $M"

# ---- install
say "installing to $DEST"
sudo mkdir -p $DEST && sudo chown "$USER_NAME" $DEST
cp -r "$KIT"/{bin,lib,serve,tools,data,gguf-py,VERSIONS} $DEST/
[ -x $DEST/venv/bin/python ] || python3 -m venv $DEST/venv
timeout 600 $DEST/venv/bin/pip install -q -r "$KIT/s2-requirements.txt"
export LD_LIBRARY_PATH=$DEST/lib STRATA_GGUF_PY=$DEST/gguf-py

# ---- pack (two seconds; the experts stay in the GGUF)
timeout 300 $DEST/venv/bin/python $DEST/tools/iq_pack.py --gguf "$M" --out $DEST/pack --compat-bf16 | tail -2

# ---- the box's own settings: performance governor and GPU persistence mode (undone by --undo)
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo none)
PM=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)
[ -f $DEST/undo.env ] || printf 'GOVERNOR=%s\nPERSISTENCE=%s\n' "$GOV" "$PM" > $DEST/undo.env
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -e "$g" ] && echo performance | sudo tee "$g" >/dev/null; done
sudo nvidia-smi -pm 1 >/dev/null 2>&1 || say "note: persistence mode not supported here"

# ---- tune: one decode benchmark per candidate, each a fresh engine start at the production context
BASE=(--pack $DEST/pack --native "$M" --expert-profile $DEST/data/expert-profile-qwen36.bin --expert-cache auto
      --prefill auto --spec-min-p 0.5 --mtp "$M" --mtp-draft-vocab $DEST/data/draft_vocab.bin
      --max-context $CTX --kv int8 --kv-grow)   # --kv-grow: the K/V takes VRAM only as far as a request reaches
PROMPT=$($DEST/venv/bin/python -c '
import sys; sys.path.insert(0, "'$DEST'/tools")
import strata_tokenizer as ST
t = ST.Tokenizer.from_gguf(sys.argv[1])
msg = "Write a Python function that checks whether a number is prime, then explain how it works."
print(",".join(map(str, t.encode("<|im_start|>user\n" + msg + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True))))' "$M")
bench() {   # bench "<env>" <engine args...> -> tok/s (0 when it does not start)
  local envs=$1; shift
  printf 'GEN 192 %s\nQUIT\n' "$PROMPT" | timeout 180 env $envs $DEST/bin/strata --serve "${BASE[@]}" "$@" 2>&1 |
    grep -oE 'generated in [0-9.]+ ms \([0-9.]+ tok/s\)' | grep -oE '[0-9.]+ tok/s' | cut -d' ' -f1 | head -1 || true
}
best=0; BEST_ARGS=(--spec 3); BEST_ENV=""
try() {   # try "<label>" "<env>" <args...>
  local label=$1 envs=$2; shift 2
  local r; r=$(bench "$envs" "$@"); r=${r:-0}
  say "  $label: $r tok/s"
  echo "$label,$r" >> $DEST/tune.csv
  if awk "BEGIN{exit !($r > $best)}"; then best=$r; BEST_ARGS=("$@"); BEST_ENV=$envs; fi
}
say "tuning (one engine start per line, ~40 s each, 180 s at most)"
: > $DEST/tune.csv
for k in 2 3 4; do try "spec $k" "" --spec $k; done
K=("${BEST_ARGS[@]}")
for r in 450 250; do try "${K[*]}, vram reserve $r MiB" "" "${K[@]}" --vram-reserve-mib $r; done
B=("${BEST_ARGS[@]}")
for w in 5 4; do try "${B[*]}, $w pool workers" "" "${B[@]}" --pool-workers $w; done
B=("${BEST_ARGS[@]}")
for e in 6 7; do try "${B[*]}, expert layout $e" "STRATA_EXP_MODE=$e" "${B[@]}"; done
say "best: ${BEST_ARGS[*]} ${BEST_ENV} -> $best tok/s"

# ---- config and service
$DEST/venv/bin/python - "$DEST" "$M" "$HOST" "$BEST_ENV" "${BASE[@]}" "${BEST_ARGS[@]}" <<'EOF'
import json, sys
dest, model, host, env, *args = sys.argv[1:]
cfg = {"exe": dest + "/bin/strata", "cwd": dest, "tokenizer": dest + "/pack/tokenizer", "model_name": "qwen3.6-35b-a3b",
       "host": host, "args": args}
json.dump(cfg, open(dest + "/strata.json", "w"), indent=1)
open(dest + "/engine.env", "w").write("LD_LIBRARY_PATH=%s/lib\n%s\n" % (dest, env.replace(" ", "\n")))
EOF
sudo tee /etc/systemd/system/strata.service >/dev/null <<EOF
[Unit]
Description=Spillway: Qwen3.6-35B-A3B (OpenAI API on $HOST:8095)
After=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$DEST
EnvironmentFile=$DEST/engine.env
ExecStart=$DEST/venv/bin/python $DEST/serve/server.py --engine strata --config $DEST/strata.json --port 8095
LimitMEMLOCK=infinity
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now strata.service

# ---- check through the API
say "waiting for the service"
for i in $(seq 120); do curl -s -m 2 localhost:8095/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s -m 120 localhost:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"Write a haiku about a quiet server room."}],"max_tokens":1500}' |
  $DEST/venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); t=r["timings"]
m = r["choices"][0]["message"]; print(m.get("content") or "(still thinking at max_tokens: %d reasoning chars)" % len(m.get("reasoning_content") or "")); print("decode %.1f tok/s, prompt %.1f tok/s, drafts %s/%s" % (t["predicted_per_second"], t["prompt_per_second"], t.get("draft_n_accepted"), t.get("draft_n")))'
# ---- the context target: one ~117K-token request (the prompt read and decode at the far end of the K/V)
say "long-context check (~117K tokens; reading the prompt takes a few minutes)"
$DEST/venv/bin/python -c '
import json
rec = "\n".join(f"Record {i}: sensor {i % 97} read {i * 7 % 1000} units at minute {i % 1440}." for i in range(4900))
q = "\n\nWhich sensor did record 4321 use? Answer with the number only."
print(json.dumps({"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": rec + q}], "max_tokens": 512}))' |
  curl -s -m 1200 localhost:8095/v1/chat/completions -H 'Content-Type: application/json' -d @- |
  $DEST/venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); t=r["timings"]; m=r["choices"][0]["message"]
print("answer: %r (expected 53)" % (m.get("content") or "(still thinking at max_tokens)")); print("prompt %d tokens at %.1f tok/s, decode %.1f tok/s, drafts %s/%s" % (r["usage"]["prompt_tokens"], t["prompt_per_second"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n")))'
say "done: strata.service is running (journalctl -u strata -f); tuning results in $DEST/tune.csv"
