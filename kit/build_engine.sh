#!/usr/bin/env bash
# Builds engine/ (the Strata copy) in the kit/Dockerfile image, for the box (sm_75) and the dev PC's 3080 (sm_86):
#   docker run --rm -v <repo>:/src -v strata-work:/work strata-build bash /src/kit/build_engine.sh
# TARGETS="strata native_grouped_parity ..." builds parity tests too.  Output: /src/out/engine/<target>. ggml comes from the engine's pinned upstream commit (third_party/ggml/VERSION.txt).
set -euo pipefail
cmake -S /src/engine -B /work/engine-build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES="75;86" -DSTRATA_PORTABLE=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc"
TARGETS=${TARGETS:-strata}
cmake --build /work/engine-build -j "$(nproc)" --target $TARGETS
mkdir -p /src/out/engine
for t in $TARGETS; do cp /work/engine-build/$t /src/out/engine/; done
ldd /src/out/engine/strata
