#!/usr/bin/env bash
# Spillway scouting: measures the laptop before tuning (docs/design/plan.md section 5). Build with dev/scout/build.sh, copy out/scout/ to the laptop and run:
#   bash scout/scout.sh
# Downloads the model into the folder if it isn't there, runs steps 1-6, and packs scout-<host>-<date>.tar.gz here.
# sudo (asked once) adds the GPU clock-lock and huge-page tests; both are undone at the end.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=$HERE
FILE=Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf
M=$HERE/$FILE
SIZE=13609247424
# llama.cpp's VRAM besides routed experts at -c 8448: dense 936 + KV 165 + recurrent 63 + compute 509 MiB (dry run)
NONCACHE_MIB=${NONCACHE_MIB:-1700}
LAYER_EXP_MIB=282   # routed experts per layer: 11280 MiB / 40

DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
[ "${DRV%%.*}" -ge 525 ] 2>/dev/null || { echo "need NVIDIA driver 525+, have '$DRV'"; exit 1; }
[ "$(stat -c %s "$M" 2>/dev/null)" = $SIZE ] \
  || curl -fL -C - -o "$M" "https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/$FILE" || exit 1
SUDO=sudo
sudo -v || SUDO=
NVSMI=$(command -v nvidia-smi)   # full path: sudo's PATH may not contain it
NC=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
SCOUT=$KIT/bin/scout
BENCH=$KIT/bin/llama-batched-bench
NAME=scout-$(hostname -s)-$(date +%Y%m%d-%H%M)
OUT=$HERE/$NAME
mkdir -p "$OUT"
exec > >(tee -a "$OUT/scout.log") 2>&1
T0=$(date +%s)
say() { echo "[$(( $(date +%s) - T0 ))s] $*"; }
step() { local id=$1; shift; say "step $id: $*"; "$@" > "$OUT/step$id.txt" 2>&1; say "step $id done (rc $?)"; }
{ cat "$KIT/VERSIONS"; echo "driver=$DRV nc=$NC noncache_mib=$NONCACHE_MIB sudo=${SUDO:-no} start_unix=$T0"; } > "$OUT/env.txt"

# 1 Hz loggers for the whole session: GPU clocks/power/throttle reasons, per-core MHz, temperatures, package energy
REASONS=clocks_event_reasons.active
nvidia-smi --query-gpu=$REASONS --format=csv >/dev/null 2>&1 || REASONS=clocks_throttle_reasons.active
nvidia-smi --query-gpu=timestamp,pstate,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,utilization.gpu,memory.used,pcie.link.gen.current,$REASONS \
  --format=csv -lms 1000 > "$OUT/gpu_1hz.csv" 2>&1 &
PIDS=$!
cpulog() {
  local n c z
  shopt -s nullglob
  n=$(nproc)
  printf 'unix'; for c in $(seq 0 $((n - 1))); do printf ',mhz%d' "$c"; done
  for z in /sys/class/thermal/thermal_zone*; do printf ',%s_c' "$(cat "$z/type")"; done
  printf ',rapl_pkg_uj\n'
  while :; do
    printf '%s' "$(date +%s)"
    for c in $(seq 0 $((n - 1))); do
      printf ',%s' $(( $(cat /sys/devices/system/cpu/cpu$c/cpufreq/scaling_cur_freq 2>/dev/null || echo 0) / 1000 )); done
    for z in /sys/class/thermal/thermal_zone*; do printf ',%s' $(( $(cat "$z/temp" 2>/dev/null || echo 0) / 1000 )); done
    printf ',%s\n' "$(cat /sys/class/powercap/intel-rapl:0/energy_uj 2>/dev/null || echo NA)"
    sleep 1
  done
}
cpulog > "$OUT/cpu_1hz.csv" &
PIDS="$PIDS $!"
[ -n "$SUDO" ] && command -v turbostat >/dev/null \
  && $SUDO turbostat --quiet --interval 1 --show Busy%,Bzy_MHz,PkgTmp,PkgWatt,CorWatt > "$OUT/turbostat.log" 2>&1 &

H2=/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
H1=/sys/kernel/mm/hugepages/hugepages-1048576kB/nr_hugepages
O2=$(cat $H2 2>/dev/null); O1=$(cat $H1 2>/dev/null)
cleanup() {   # undo everything: loggers, GPU clock lock, huge-page reservations
  kill $PIDS 2>/dev/null
  if [ -n "$SUDO" ]; then
    $SUDO pkill -x turbostat 2>/dev/null
    $SUDO "$NVSMI" -rgc >/dev/null 2>&1
    [ -n "$O2" ] && $SUDO sh -c "echo $O2 > $H2"
    [ -n "$O1" ] && $SUDO sh -c "echo $O1 > $H1"
  fi
}
trap cleanup EXIT

# 1: inventory
step 1 $SUDO env "PATH=$PATH" bash "$KIT/probe.sh"

# 2: micro-benchmarks (context and free VRAM, GPU read rate, launches, PCIe, doorbell, DDR4)
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > "$OUT/vram_before_context.csv" 2>&1
step 2 "$SCOUT" info gpu pcie ram

# 3: ggml's CPU expert kernels
step 3 "$SCOUT" cpuq

# 4: llama.cpp baseline on the real file; batched-bench skips --fit, so both configurations are sized from step 2
ARGS=(-m "$M" -c 8448 -npp 2048,8192 -ntg 128 -npl 1 -t "$NC" -ngl 99)
FREE=$(grep -o 'free_after_context_mib=[0-9]*' "$OUT/step2.txt" | cut -d= -f2)
GPU_LAYERS=$(( (${FREE:-7800} - NONCACHE_MIB - 128) / LAYER_EXP_MIB ))
NCMOE=$(( 40 - (GPU_LAYERS < 0 ? 0 : GPU_LAYERS > 40 ? 40 : GPU_LAYERS) ))
step 4a "$BENCH" "${ARGS[@]}" -ncmoe "$NCMOE"
CACHE=$(( ${FREE:-7800} - NONCACHE_MIB ))
step 4b "$BENCH" "${ARGS[@]}" -cmoe --moe-cache-mib "$CACHE"
grep -qiE 'out of memory|failed to allocate|unable to allocate' "$OUT/step4b.txt" \
  && say "4b ran out of VRAM: rerun with NONCACHE_MIB=2100 bash scout.sh"

# 5: GPU read loop + CPU expert loop together for 2 min; with sudo, the last 60 s with the GPU core clock locked low
say "step 5: combined load, 120 s"
"$SCOUT" combined 120 > "$OUT/step5.txt" 2>&1 &
P=$!
if [ -n "$SUDO" ]; then
  sleep 60
  echo "lgc_unix=$(date +%s)" >> "$OUT/step5-lgc.txt"
  $SUDO "$NVSMI" -lgc 300,900 >> "$OUT/step5-lgc.txt" 2>&1
fi
wait $P
[ -n "$SUDO" ] && $SUDO "$NVSMI" -rgc >> "$OUT/step5-lgc.txt" 2>&1
say "step 5 done"

# 6: page sizes (sudo only): reserve 2 MB and 1 GB pages at runtime, measure, release
if [ -n "$SUDO" ]; then
  $SUDO sh -c "sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory;
               echo $(( ${O2:-0} + 1024 )) > $H2; [ -e $H1 ] && echo $(( ${O1:-0} + 2 )) > $H1"
  echo "reserved 2m=$(cat $H2) 1g=$(cat $H1 2>/dev/null)" > "$OUT/step6-reserved.txt"
  step 6 "$SCOUT" hugepages 2048
fi

cleanup; trap - EXIT
say "total $(( $(date +%s) - T0 ))s"
tar -C "$HERE" -czf "$HERE/$NAME.tar.gz" "$NAME"
echo "Bring back: $HERE/$NAME.tar.gz"
