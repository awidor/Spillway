#!/usr/bin/env bash
# Spillway: Qwen3.6-35B-A3B on an RTX 2070 Max-Q laptop (8 GB, Turing, 16 GB RAM, Linux), as a systemd service with an
# OpenAI-compatible API. Installs the engine, packs the model, measures the settings only this machine can decide,
# starts spillway.service and checks it through the API.
#   curl -fsSL https://github.com/awidor/Spillway/releases/latest/download/install.sh | bash
#   bash install.sh [--model PATH] [--port N] [--lan | --local]   install or update, tune, start (asks for sudo once)
#                                     --lan: listen on the network, with an API key; --local: this machine only
#   bash install.sh --retune                              measure the settings again with the installed files
#   bash install.sh --uninstall                           remove the service and /opt/spillway (the model is kept)
# Run outside an unpacked release, it downloads the release (SPILLWAY_VERSION=v0.5.0 for one version) and runs that.
set -eu
REPO=awidor/Spillway
DEST=/opt/spillway
USER_NAME=${SUDO_USER:-$(id -un)}
FILE=Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf
SIZE=13609247424
URL=https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/$FILE
CACHE=${XDG_CACHE_HOME:-$HOME/.cache}/spillway
CTX=131072   # the target: at least 128K tokens of context
say() { echo "[$(date +%T)] $*"; }
die() { echo "spillway: $*" >&2; exit 1; }
SRC=${BASH_SOURCE[0]:-}
KIT=$( [ -n "$SRC" ] && cd "$(dirname "$SRC")" && pwd || pwd)

# ---- bootstrap: not inside an unpacked release (curl | bash, or the file alone): fetch the release and run its installer
if [ ! -x "$KIT/bin/strata" ]; then
  V=${SPILLWAY_VERSION:-}
  BASE=https://github.com/$REPO/releases/${V:+download/$V}; [ -n "$V" ] || BASE=https://github.com/$REPO/releases/latest/download
  T=spillway-linux-x86_64.tar.gz
  mkdir -p "$CACHE" && cd "$CACHE"
  say "downloading $T ${V:-(latest release)}"
  curl -fL -o "$T" "$BASE/$T" && curl -fsSL -o "$T.sha256" "$BASE/$T.sha256" || die "download failed"
  sha256sum -c "$T.sha256" >/dev/null || die "$T does not match its checksum; delete $CACHE/$T and run again"
  rm -rf "$CACHE/spillway" && tar -xzf "$T" && rm -f "$T"
  exec bash "$CACHE/spillway/install.sh" "$@"
fi

MODE=install MODEL= ARG_MODEL=
[ -f $DEST/settings.env ] && . $DEST/settings.env   # PORT, HOST, API_KEY, MODEL of the installed copy
PORT=${PORT:-8095} HOST=${HOST:-127.0.0.1} API_KEY=${API_KEY:-}
while [ $# -gt 0 ]; do
  case $1 in
    --retune) MODE=retune ;;
    --uninstall|--undo) MODE=uninstall ;;
    --model) ARG_MODEL=$(readlink -f "$2"); shift ;;
    --port) PORT=$2; shift ;;
    --lan) HOST=0.0.0.0 ;;
    --local) HOST=127.0.0.1 ;;
    -h|--help) sed -n 2,10p "$KIT/install.sh"; exit 0 ;;
    *) die "unknown option $1 (--help lists them)" ;;
  esac
  shift
done

restore_os() {   # the governor and persistence mode from before the first install
  [ -f "$1" ] || return 0
  local GOVERNOR PERSISTENCE; . "$1"
  for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do [ -e "$g" ] && echo "$GOVERNOR" | sudo tee "$g" >/dev/null; done
  [ "$PERSISTENCE" = Disabled ] && sudo nvidia-smi -pm 0 >/dev/null || true
}
remove_unit() {
  sudo systemctl disable --now "$1" 2>/dev/null || true
  sudo rm -f "/etc/systemd/system/$1" && sudo systemctl daemon-reload
}

if [ $MODE = uninstall ]; then
  remove_unit spillway.service
  restore_os $DEST/undo.env
  sudo rm -rf $DEST /usr/local/bin/spillway
  say "removed: spillway.service, $DEST and the spillway command; governor and persistence mode restored"
  [ -n "$MODEL" ] && say "the model is kept: $MODEL"
  exit 0
fi

# ---- checks: the laptop's configuration, nothing else
[ "$(uname -sm)" = "Linux x86_64" ] || die "needs Linux on x86_64"
grep -qi microsoft /proc/version 2>/dev/null && die "WSL is not supported: run it on the laptop's own Linux"
command -v systemctl >/dev/null && [ -d /run/systemd/system ] || die "needs systemd"
command -v nvidia-smi >/dev/null || die "no NVIDIA driver (nvidia-smi not found)"
IFS=, read -r GPU DRV CC VRAM < <(nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader,nounits | head -1)
GPU=$(echo $GPU) VRAM=$(echo $VRAM) CC=$(echo $CC)
[ "${DRV%%.*}" -ge 525 ] 2>/dev/null || die "needs NVIDIA driver 525 or newer, have $DRV"
[ "$CC" = 7.5 ] && [ "$VRAM" -ge 7800 ] || die "built for the RTX 2070 Max-Q (Turing, 8 GB); this is a $GPU ($CC, $VRAM MiB)"
case $GPU in *2070*Max-Q*) ;; *) say "note: tuned for the RTX 2070 Max-Q; this $GPU is a Turing card too, but untested" ;; esac
[ "$(awk '/MemTotal/ {print int($2 / 1048576)}' /proc/meminfo)" -ge 15 ] || die "needs 16 GB of RAM"
command -v python3 >/dev/null && python3 -c 'import sys; sys.exit(sys.version_info < (3, 10))' || die "needs Python 3.10+ (python3)"

# ---- the model: --model, the installed copy's, next to this kit, the Hugging Face cache, a v0.4 install, else download
OLD=/opt/strata/strata.json   # releases up to v0.4.0 installed to /opt/strata and kept the model path in its config
[ -f $OLD ] && OLDM=$(python3 -c 'import json,sys; a=json.load(open(sys.argv[1]))["args"]; print(a[a.index("--native")+1])' $OLD 2>/dev/null) || OLDM=
M=
for c in "$ARG_MODEL" "$MODEL" "$KIT/$FILE" "$CACHE/$FILE" ~/.cache/huggingface/hub/models--byteshape--Qwen3.6-35B-A3B-MTP-GGUF/snapshots/*/$FILE "$OLDM"; do
  [ -n "$c" ] && [ "$(stat -L -c %s "$c" 2>/dev/null)" = $SIZE ] && { M=$(readlink -f "$c"); break; }
done
if [ -z "$M" ]; then
  [ -n "$ARG_MODEL" ] && die "$ARG_MODEL is not byteshape's $FILE ($SIZE bytes)"
  mkdir -p "$CACHE"
  [ "$(df -B1 --output=avail "$CACHE" | tail -1)" -gt $((SIZE + 500000000)) ] || die "needs 14 GB free in $CACHE for the model"
  M=$CACHE/$FILE; say "downloading the model (13.6 GB) to $M"; curl -fL -C - -o "$M" "$URL"
fi
say "model: $M"

[ $MODE = install ] && [ "$KIT" = $DEST ] && die "this is the installed copy: use spillway update or spillway retune"
if [ $MODE = install ]; then
  echo "This will:
  - install Spillway to $DEST and the 'spillway' command to /usr/local/bin
  - create spillway.service: starts at boot, API and dashboard on http://$HOST:$PORT
  - set the CPU governor to performance and turn on GPU persistence mode each time the service starts
    (restored by 'spillway uninstall')
  - measure about 15 settings on this machine, then check a ~117K-token request"
  [ -f /etc/systemd/system/strata.service ] && grep -q /opt/strata/serve/server.py /etc/systemd/system/strata.service &&
    echo "  - replace the v0.4 install (strata.service, /opt/strata)"
fi
sudo true   # asks for the password once, here (sudo -v asks even with NOPASSWD)

# ---- install (or update) the files
if [ $MODE = install ]; then
  python3 -c 'import venv, ensurepip' 2>/dev/null || { say "installing python3-venv"; timeout 300 sudo apt-get install -y python3-venv; }
  if grep -qs /opt/strata/serve/server.py /etc/systemd/system/strata.service; then
    say "removing the v0.4 install"
    remove_unit strata.service
    sudo mkdir -p $DEST && [ -f /opt/strata/undo.env ] && sudo cp -n /opt/strata/undo.env $DEST/
    sudo rm -rf /opt/strata
  fi
  sudo systemctl stop spillway.service 2>/dev/null || true
  say "installing to $DEST"
  sudo mkdir -p $DEST && sudo chown -R "$USER_NAME" $DEST
  rm -rf $DEST/{bin,lib,serve,tools,data,gguf-py}
  cp -r "$KIT"/{bin,lib,serve,tools,data,gguf-py,VERSIONS,install.sh,spillway,chat.py,requirements.txt} $DEST/
  [ -x $DEST/venv/bin/python ] || python3 -m venv $DEST/venv
  timeout 600 $DEST/venv/bin/pip install -q -r $DEST/requirements.txt
  sudo ln -sf $DEST/spillway /usr/local/bin/spillway
  # the pack: two seconds; the experts stay in the GGUF
  LD_LIBRARY_PATH=$DEST/lib STRATA_GGUF_PY=$DEST/gguf-py timeout 300 \
    $DEST/venv/bin/python $DEST/tools/iq_pack.py --gguf "$M" --out $DEST/pack --compat-bf16 | tail -2
else
  [ -x $DEST/bin/strata ] || die "not installed: run install.sh first"
  sudo systemctl stop spillway.service
fi
if [ $HOST != 127.0.0.1 ] && [ -z "$API_KEY" ]; then API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))'); fi
printf 'PORT=%s\nHOST=%s\nAPI_KEY=%s\nMODEL=%s\n' "$PORT" "$HOST" "$API_KEY" "$M" > $DEST/settings.env
chmod 600 $DEST/settings.env

# ---- the OS settings the measurements assume (the service sets them again on every start)
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo none)
PM=$(nvidia-smi --query-gpu=persistence_mode --format=csv,noheader | head -1)
[ -f $DEST/undo.env ] || printf 'GOVERNOR=%s\nPERSISTENCE=%s\n' "$GOV" "$PM" > $DEST/undo.env
sudo $DEST/spillway boost

# ---- tune: one decode benchmark per candidate, each a fresh engine start at the production context
export LD_LIBRARY_PATH=$DEST/lib STRATA_GGUF_PY=$DEST/gguf-py
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
    tee -a $DEST/tune.log | grep -oE 'generated in [0-9.]+ ms \([0-9.]+ tok/s\)' | grep -oE '[0-9.]+ tok/s' | cut -d' ' -f1 | head -1 || true
}
best=0; BEST_ARGS=(--spec 3); BEST_ENV=""
try() {   # try "<label>" "<env>" <args...>
  local label=$1 envs=$2; shift 2
  local r; r=$(bench "$envs" "$@"); r=${r:-0}
  say "  $label: $r tok/s   (${*:-} ${envs:-})"
  echo "$label,$r" >> $DEST/tune.csv
  if awk "BEGIN{exit !($r > $best)}"; then best=$r; BEST_ARGS=("$@"); BEST_ENV=$envs; fi
}
say "measuring settings (one engine start per line, ~40 s each, 180 s at most)"
: > $DEST/tune.csv; : > $DEST/tune.log
for k in 2 3 4; do try "draft length $k" "" --spec $k; done
K=("${BEST_ARGS[@]}")
for r in 450 250; do try "vram reserve $r MiB" "" "${K[@]}" --vram-reserve-mib $r; done
B=("${BEST_ARGS[@]}")
# the share of cache misses read over PCIe instead of computed on the CPU (default: the engine's link-based guess;
# a desktop 2070 SUPER on PCIe 3.0 with a 2-core CPU decoded 79 tok/s at 1 against 59 at the guess)
for f in 0.66 1; do try "PCIe share $f" "" "${B[@]}" --pcie-frac $f; done
B=("${BEST_ARGS[@]}")
# the GPU prefetch of the next layer's predicted misses: on by default with up to 3 a layer (it shares the link with
# the window's own copies and competes with the CPU's experts for RAM), and the copy kernel's blocks (default 4)
try "no GPU prefetch" "STRATA_PREFETCH=0" "${B[@]}"
for p in 2 5; do try "GPU prefetch up to $p a layer" "STRATA_PREFETCH_MAX=$p" "${B[@]}"; done
B=("${BEST_ARGS[@]}"); E=$BEST_ENV
try "8 copy blocks" "$E STRATA_FETCH_FORK=8" "${B[@]}"
B=("${BEST_ARGS[@]}"); E=$BEST_ENV
for w in 5 4; do try "$w pool workers" "$E" "${B[@]}" --pool-workers $w; done
B=("${BEST_ARGS[@]}"); E=$BEST_ENV
for e in 6 7; do try "expert layout $e" "$E STRATA_EXP_MODE=$e" "${B[@]}"; done
say "best: $(echo ${BEST_ARGS[*]} $BEST_ENV) -> $best tok/s"
[ "$best" != 0 ] || { tail -5 $DEST/tune.log; die "the engine did not start (above; the whole log: $DEST/tune.log)"; }
echo "best,$(echo ${BEST_ARGS[*]} $BEST_ENV),$best" >> $DEST/tune.csv

# ---- config and service
$DEST/venv/bin/python - "$DEST" "$HOST" "$API_KEY" "$BEST_ENV" "${BASE[@]}" "${BEST_ARGS[@]}" <<'EOF'
import json, os, sys
dest, host, key, env, *args = sys.argv[1:]
cfg = {"exe": dest + "/bin/strata", "cwd": dest, "tokenizer": dest + "/pack/tokenizer", "model_name": "qwen3.6-35b-a3b",
       "host": host, "args": args}
if key:
    cfg["api_key"] = key
with open(os.open(dest + "/strata.json", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
    json.dump(cfg, f, indent=1)
open(dest + "/engine.env", "w").write("LD_LIBRARY_PATH=%s/lib\n%s\n" % (dest, "\n".join(env.split())))
EOF
sudo tee /etc/systemd/system/spillway.service >/dev/null <<EOF
[Unit]
Description=Spillway: Qwen3.6-35B-A3B (OpenAI API on $HOST:$PORT)
After=network-online.target

[Service]
User=$USER_NAME
WorkingDirectory=$DEST
EnvironmentFile=$DEST/engine.env
ExecStartPre=+$DEST/spillway boost
ExecStart=$DEST/venv/bin/python $DEST/serve/server.py --engine strata --config $DEST/strata.json --port $PORT
LimitMEMLOCK=infinity
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now spillway.service

# ---- check through the API
API=http://127.0.0.1:$PORT
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
say "waiting for the service"
for i in $(seq 120); do curl -s -m 2 "${AUTH[@]}" $API/v1/models >/dev/null 2>&1 && break; sleep 2; done
curl -s -m 120 "${AUTH[@]}" $API/v1/chat/completions -H 'Content-Type: application/json' \
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
  curl -s -m 1200 "${AUTH[@]}" $API/v1/chat/completions -H 'Content-Type: application/json' -d @- |
  $DEST/venv/bin/python -c 'import json,sys; r=json.load(sys.stdin); t=r["timings"]; m=r["choices"][0]["message"]
print("answer: %r (expected 53)" % (m.get("content") or "(still thinking at max_tokens)")); print("prompt %d tokens at %.1f tok/s, decode %.1f tok/s, drafts %s/%s" % (r["usage"]["prompt_tokens"], t["prompt_per_second"], t["predicted_per_second"], t.get("draft_n_accepted"), t.get("draft_n")))'

LAN=; [ $HOST != 127.0.0.1 ] && LAN=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "
Spillway is running.
  Dashboard:            http://${LAN:-127.0.0.1}:$PORT/${API_KEY:+#key=$API_KEY}
  OpenAI API:           http://${LAN:-127.0.0.1}:$PORT/v1   (model: qwen3.6-35b-a3b)${API_KEY:+
  API key:              $API_KEY   (also in $DEST/settings.env)}
  In the terminal:      spillway chat
  Status, logs:         spillway status, spillway logs
  Remove:               spillway uninstall"
