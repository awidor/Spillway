#!/usr/bin/env bash
# Builds engine/ (the Strata copy) in the kit/Dockerfile image for the laptop's RTX 2070 Max-Q (sm_75); ./dev.sh build runs it:
#   docker run --rm -v <repo>:/src -v spillway-work:/work spillway-build bash /src/kit/build_engine.sh
# TARGETS="strata native_grouped_parity ..." builds parity tests too; CUDA_ARCHS overrides sm_75.  Output: /src/out/engine/<target>.
# ggml comes from the engine's pinned upstream commit (third_party/ggml/VERSION.txt).
set -euo pipefail
cmake -S /src/engine -B /work/engine-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75}" -DSTRATA_PORTABLE=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
TARGETS=${TARGETS:-strata}
cmake --build /work/engine-build -j "$(nproc)" --target $TARGETS
mkdir -p /src/out/engine
for t in $TARGETS; do cp /work/engine-build/$t /src/out/engine/; done
ldd /src/out/engine/strata
