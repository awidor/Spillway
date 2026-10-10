#!/usr/bin/env bash
# Assembles out/s2kit (the S2 folder for the box) from the engine build, in the kit/Dockerfile image (patchelf):
#   docker run --rm -v <repo>:/src strata-build bash /src/kit/package_s2.sh
# Needs out/engine/strata (kit/build_engine.sh) and out/s1kit/lib (kit/build.sh: the same CUDA 12.8 libraries).
set -euo pipefail
E=/src/engine K=/src/out/s2kit
rm -rf $K && mkdir -p $K/bin $K/lib $K/tools $K/data
cp /src/out/engine/strata $K/bin/ && patchelf --set-rpath '$ORIGIN/../lib' $K/bin/strata
cp /src/out/s1kit/lib/* $K/lib/
cp -r $E/serve $K/ && rm -rf $K/serve/__pycache__ $K/serve/test_*.py $K/serve/fixtures
cp $E/tools/{iq_pack.py,strata_tokenizer.py,gguf_reader.py,_paths.py} $K/tools/
cp $E/data/{expert-profile-qwen36.bin,draft_vocab.bin} $K/data/
cp -r $E/third_party/llama.cpp/gguf-py $K/gguf-py && rm -rf $K/gguf-py/tests
cp /src/kit/s2.sh /src/kit/s2-requirements.txt $K/
find $K -name '*.sh' -o -name '*.py' -o -name '*.txt' | xargs sed -i 's/\r$//'
{ echo "engine: Strata 650cf81 + Spillway changes (RESULTS.md); built $(date -u)"; nvcc --version | tail -1; } > $K/VERSIONS
ldd $K/bin/strata | grep -E "not found" && { echo "unbundled dependency"; exit 1; }
du -sh $K
