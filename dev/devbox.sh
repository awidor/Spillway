#!/usr/bin/env bash
# The dev server: a desktop RTX 2070 SUPER (Turing, sm_75, like the laptop's 2070 Max-Q) on a Proxmox host, reached over
# ssh. Builds and runs happen in a long-lived container there (the kit/Dockerfile image, GPU attached); this PC only
# edits and syncs. SYS_ADMIN in the container lets Nsight Compute (ncu) read the GPU's counters.
#   bash dev/devbox.sh setup            the container (once; needs the spillway-build image on the server)
#   bash dev/devbox.sh sync             this tree -> /root/strata/src (no out/, no .git)
#   bash dev/devbox.sh build [targets]  sync, then kit/build_engine.sh in the container (default: strata)
#   bash dev/devbox.sh run '<cmd>'      sync, then <cmd> in the container (cwd /src; /strata/{models,packs})
# Env: DEVBOX=user@host (required): the server, reached with ssh keys.
set -euo pipefail
H=${DEVBOX:?set DEVBOX=user@host, the GPU server}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SSH=(ssh -o BatchMode=yes -c aes128-gcm@openssh.com "$H")
sync() {
    tar -C "$ROOT" --exclude=./out --exclude=.git --exclude=__pycache__ --exclude=./.claude \
        --exclude='./LLM-Laptop-Hardware-*' -cf - . | "${SSH[@]}" 'tar -C /root/strata/src -xf -'   # mtimes kept: ninja rebuilds only what changed
}
case "${1:-}" in
    setup)
        # its own slice, 16 GB like the laptop (the host's docker.slice caps the other containers at 6 GB)
        "${SSH[@]}" 'printf "[Unit]\nDescription=Spillway dev container (16 GB, the laptop RAM)\nBefore=slices.target\n[Slice]\nMemoryAccounting=yes\nMemoryMax=16G\n" > /etc/systemd/system/strata.slice &&
            systemctl daemon-reload && systemctl start strata.slice &&
            docker rm -f spillway-dev >/dev/null 2>&1; mkdir -p /root/strata/{src,work,models,packs,out} &&
            docker run -d --name spillway-dev --restart unless-stopped --cgroup-parent=strata.slice --gpus all --ipc=host --ulimit memlock=-1:-1 \
              --cap-add SYS_ADMIN \
              -v /root/strata/src:/src -v /root/strata/work:/work -v /root/strata/out:/src/out -v /root/strata:/strata \
              spillway-build sleep infinity >/dev/null && docker exec spillway-dev nvidia-smi -L' ;;
    sync) sync ;;
    build)
        shift; sync
        "${SSH[@]}" "docker exec -e TARGETS='${*:-strata}' spillway-dev bash /src/kit/build_engine.sh 2>&1 | grep -E 'error|FAILED|Linking' | tail -20" ;;
    run)
        shift; sync
        "${SSH[@]}" "docker exec -w /src spillway-dev bash -c $(printf %q "$*")" ;;
    *) sed -n 2,10p "$0"; exit 1 ;;
esac
