#!/usr/bin/env bash
# S1 item 1 (PLAN.md section 5): read-only inventory of the dedicated, headless box.
# Changes nothing and adds no load; runs in seconds. Serial numbers and UUIDs are filtered out.
#   bash probe.sh > probe.txt 2>&1            (root-only parts are skipped)
#   sudo bash probe.sh > probe.txt 2>&1       (adds dmidecode, full lspci, RAPL energy)
set -u
nosec() { grep -viE 'serial|uuid'; }
sec() { echo; echo "== $* =="; }

sec "system"
for f in sys_vendor product_name product_version bios_vendor bios_version bios_date board_name; do
  printf '%s: %s\n' "$f" "$(cat /sys/class/dmi/id/$f 2>/dev/null)"; done
uname -r; (. /etc/os-release && echo "$PRETTY_NAME"); systemctl get-default; uptime

sec "kernel options the plan wants (2b)"
cat /proc/cmdline
grep -E 'CONFIG_NO_HZ_FULL|CONFIG_HUGETLBFS|CONFIG_TRANSPARENT_HUGEPAGE=|CONFIG_RCU_NOCB_CPU' "/boot/config-$(uname -r)" 2>/dev/null \
  || echo "no /boot/config for this kernel"
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled) / defrag $(cat /sys/kernel/mm/transparent_hugepage/defrag)"
grep -i huge /proc/meminfo
grep -o -m1 pdpe1gb /proc/cpuinfo || echo "no 1G pages (pdpe1gb)"
echo "ASPM policy: $(cat /sys/module/pcie_aspm/parameters/policy 2>/dev/null)"
swapon --show; free -m

sec "CPU: topology, governor, turbo, power limits"
lscpu | grep -E 'Model name|Thread|Core|Socket|MHz|L1d|L2|L3|Flags' | cut -c1-200
lscpu -e
for c in 0 1 2 3 4 5; do echo "cpu$c siblings: $(cat /sys/devices/system/cpu/cpu$c/topology/thread_siblings_list)"; done
echo "driver: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null)," \
     "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)," \
     "EPP: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)," \
     "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null)"
for z in /sys/class/powercap/intel-rapl:0/constraint_*; do
  [ -e "$z"_power_limit_uw ] && echo "$(cat "$z"_name): $(( $(cat "$z"_power_limit_uw) / 1000000 )) W," \
    "window $(cat "$z"_time_window_us 2>/dev/null) us"; done
command -v turbostat >/dev/null && echo "turbostat: present" || echo "turbostat: missing (linux-tools)"

sec "temperatures at rest (S1 step 5 measures clocks under load)"
for t in /sys/class/thermal/thermal_zone*; do echo "$(cat $t/type): $(( $(cat $t/temp) / 1000 )) C"; done

sec "RAM modules (root)"
if [ "$(id -u)" = 0 ]; then dmidecode -t memory | grep -E 'Size|Speed|Rank|Part|Locator|Type:' | nosec; else echo "skipped: not root"; fi

sec "GPU"
nvidia-smi -q | nosec
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,clocks.max.memory,clocks.max.graphics,power.limit,power.max_limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,persistence_mode,compute_mode --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
gpu=$(lspci -D | awk '/NVIDIA/ && /VGA|3D/ {print $1; exit}')
if [ -n "$gpu" ]; then lspci -vvv -s "$gpu" 2>/dev/null | grep -E 'LnkCap|LnkSta|LnkCtl|ASPM|MaxPayload|MaxReadReq'; fi

sec "power profile"
echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile 2>/dev/null || echo none)" "choices: $(cat /sys/firmware/acpi/platform_profile_choices 2>/dev/null)"

sec "disk"
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN
df -h --output=source,avail,target / "$HOME" 2>/dev/null

sec "toolchain and network"
for t in gcc g++ cmake ninja nvcc python3 git curl; do printf '%s: ' "$t"; ($t --version 2>/dev/null | head -1) || echo missing; done
ls -d /usr/local/cuda* 2>/dev/null
ip -br addr | grep -v '^lo'
getent hosts huggingface.co >/dev/null && echo "DNS ok (huggingface.co resolves)" || echo "no DNS / offline"
