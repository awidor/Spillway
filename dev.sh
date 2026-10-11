#!/usr/bin/env bash
# Spillway's development commands. Needs Docker (on Windows: Docker Desktop, and run this from Git Bash). The GPU test
# needs the NVIDIA Container Toolkit and a Turing card; a remote GPU server works through dev/devbox.sh.
#   ./dev.sh setup             the build image and llama.cpp's gguf-py (once, and after kit/Dockerfile changes)
#   ./dev.sh build [targets]   the engine: out/engine/strata (name parity tests to build them too)
#   ./dev.sh package           the release: out/release/ (tarball, checksum, install.sh) and out/spillway/
#   ./dev.sh test              the server's tests (Python, no GPU)
#   ./dev.sh test-gpu [regex]  build and run the engine's tests with ctest (GPU; a regex picks some)
#   ./dev.sh shell             a shell in the build image
set -euo pipefail
export MSYS_NO_PATHCONV=1   # Git Bash: keep /src as a container path
cd "$(dirname "$0")"
ROOT=$(pwd -W 2>/dev/null || pwd)
IMG=spillway-build
RUN=(docker run --rm -v "$ROOT:/src" -v spillway-work:/work)
LLAMA=engine/third_party/llama.cpp
LLAMA_SHA=$(cut -d' ' -f1 engine/third_party/ggml/VERSION.txt)   # gguf-py from the commit the engine's ggml comes from

case "${1:-}" in
  setup)
    docker build -t $IMG kit
    if [ ! -d $LLAMA/gguf-py ]; then
      git clone -q --filter=blob:none --no-checkout https://github.com/ggml-org/llama.cpp $LLAMA
      git -C $LLAMA sparse-checkout set gguf-py && git -C $LLAMA checkout -q "$LLAMA_SHA"
    fi
    echo "ready: ./dev.sh build" ;;
  build) shift; "${RUN[@]}" -e TARGETS="${*:-strata}" $IMG bash /src/kit/build_engine.sh ;;
  package) "${RUN[@]}" -e SPILLWAY_VERSION="$(git describe --tags --always --dirty 2>/dev/null || echo dev)" $IMG bash /src/kit/package.sh ;;
  test)
    docker run --rm -v "$ROOT:/src" -w /src/engine python:3.12-slim bash -o pipefail -c \
      'pip install -q --root-user-action=ignore --disable-pip-version-check -r /src/kit/requirements.txt pillow &&
       python -m unittest discover -s serve -p "test_*.py" -t . 2>&1 | tail -5' ;;
  test-gpu)
    "${RUN[@]}" --gpus all $IMG bash -c "bash /src/kit/build_engine.sh >/dev/null && cmake --build /work/engine-build -j \$(nproc) &&
      ctest --test-dir /work/engine-build --output-on-failure ${2:+-R '$2'}" ;;
  shell) "${RUN[@]}" -it $IMG bash ;;
  *) sed -n 2,9p "$0"; exit 1 ;;
esac
