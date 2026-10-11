#!/usr/bin/env bash
# Assembles the release from the engine build, in the kit/Dockerfile image (patchelf); ./dev.sh package runs it:
#   docker run --rm -v <repo>:/src spillway-build bash /src/kit/package.sh
# Needs out/engine/strata (kit/build_engine.sh) and engine/third_party/llama.cpp/gguf-py (./dev.sh setup).
# Output: out/spillway/ and out/release/{spillway-linux-x86_64.tar.gz,.sha256,install.sh}.
set -euo pipefail
E=/src/engine K=/src/out/spillway R=/src/out/release
rm -rf $K $R && mkdir -p $K/bin $K/lib $K/tools $K/data $R
cp /src/out/engine/strata $K/bin/
# the CUDA libraries the engine loads (ldd lists them all, libcublasLt too), from this image's toolkit: the laptop has
# only the driver
for so in $(ldd $K/bin/strata | awk '/\/usr\/local\/cuda/ {print $3}' | sort -u); do cp -L "$so" $K/lib/; done
patchelf --set-rpath '$ORIGIN/../lib' $K/bin/strata
cp -r $E/serve $K/ && rm -rf $K/serve/__pycache__ $K/serve/test_*.py $K/serve/fixtures
cp $E/tools/{iq_pack.py,strata_tokenizer.py,gguf_reader.py,_paths.py} $K/tools/
cp $E/data/{expert-profile-qwen36.bin,draft_vocab.bin} $K/data/
cp -r $E/third_party/llama.cpp/gguf-py $K/gguf-py && rm -rf $K/gguf-py/tests
cp /src/kit/{install.sh,spillway,requirements.txt} $E/chat.py $K/
chmod +x $K/install.sh $K/spillway $K/bin/strata
find $K -type f \( -name '*.sh' -o -name '*.py' -o -name '*.txt' -o -name spillway \) | xargs sed -i 's/\r$//'
V=${SPILLWAY_VERSION:-dev}   # from ./dev.sh: the image's git refuses a checkout it does not own
{ echo "Spillway $V, built $(date -u +%F)"; echo "engine: Strata 650cf81 + the changes in engine/SOURCE.md"; nvcc --version | tail -1; } > $K/VERSIONS
LD_LIBRARY_PATH= ldd $K/bin/strata | grep -E "not found" && { echo "unbundled dependency"; exit 1; }
tar -C /src/out -czf $R/spillway-linux-x86_64.tar.gz spillway
(cd $R && sha256sum spillway-linux-x86_64.tar.gz > spillway-linux-x86_64.tar.gz.sha256)
cp $K/install.sh $R/
du -sh $K $R/*
