# Developing Spillway

## Layout

| Path | |
|---|---|
| `engine/` | the engine and server: a copy of Strata `650cf81`, with the changes in `engine/SOURCE.md` |
| `engine/serve/` | the HTTP server (OpenAI and Anthropic APIs, `/metrics`) and the dashboard (`web/`) |
| `kit/` | the build image, the build and packaging scripts, `install.sh` and the `spillway` command |
| `dev/` | benchmarks, A/B and parity scripts, the speed model, the remote GPU server script |
| `dev/scout/` | the scouting kit: llama.cpp baseline, hardware probe, micro-benchmarks |
| `docs/` | measurements (`BENCHMARKS.md`) and the original plan (`design/plan.md`) |

## Build

You need Docker. On Windows, use Docker Desktop and run the commands in Git Bash.

```bash
./dev.sh setup     # the build image (CUDA 12.8, Ubuntu 20.04) and llama.cpp's gguf-py at the engine's pinned commit
./dev.sh build     # out/engine/strata, for sm_75
./dev.sh package   # out/release/: spillway-linux-x86_64.tar.gz, its .sha256, install.sh
```

`./dev.sh build native_grouped_parity route_window_parity` also builds those parity tests. To build for another card
while developing, use `CUDA_ARCHS=86 ./dev.sh build`. Releases are sm_75 only.

Native build, without Docker (CUDA 12.x, gcc 13, CMake 3.31, Ninja):

```bash
cmake -S engine -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target strata
```

## Test

```bash
./dev.sh test              # the server's tests (Python 3.12 in Docker, no GPU)
./dev.sh test-gpu [regex]  # the engine's tests with ctest (NVIDIA Container Toolkit, Turing GPU)
```

Every engine change is measured on a Turing card and checked for parity before it lands. These tools do it:

| Script | |
|---|---|
| `dev/release_ab.sh OLD NEW` | an old engine build against a new one, using the laptop's flags; prints the prompt time, decode speed and a hash of the tokens |
| `dev/ab_fetch.sh "ENV=a" "ENV=b"` | A/B of one environment switch |
| `dev/bench_engine.sh` | one decode benchmark with the GPU stage profile |
| `dev/parity_greedy.py` | greedy output against llama.cpp on the same GGUF |
| `dev/*_bench.cu` | kernel micro-benchmarks (attention, IQ MMVQ, PCIe zero-copy) |

## Remote GPU server

`dev/devbox.sh` syncs this tree to a server with a Turing card and runs builds and commands in a container there.

```bash
export DEVBOX=user@host
bash dev/devbox.sh setup                          # once: the container (needs the spillway-build image there)
bash dev/devbox.sh build
bash dev/devbox.sh run 'bash dev/release_ab.sh /work/old/bin/strata /src/out/engine/strata 2'
```

## Engine switches

The installer measures the settings marked "tuned" on the laptop.

| Switch | Default | |
|---|---|---|
| `--spec N` | tuned (2-4) | MTP draft length |
| `--vram-reserve-mib N` | tuned | VRAM left free beside the expert cache |
| `--pcie-frac F` | tuned (engine guess, 0.66, 1) | share of cache misses copied over PCIe instead of computed on the CPU |
| `--pool-workers N` | tuned | CPU expert threads |
| `STRATA_PREFETCH=0`, `STRATA_PREFETCH_MAX=N` | on, 3 | GPU prefetch of the next layer's predicted misses (tuned) |
| `STRATA_FETCH_FORK=N` | 4 | blocks of the PCIe copy kernel (tuned) |
| `STRATA_EXP_MODE=N` | 8 | expert kernel layout (tuned) |
| `STRATA_ATTN_DENSE=0` | on | flash-decoding attention for the dense selection |
| `STRATA_ATTN_TC=0` | on | decode windows on tensor cores |
| `STRATA_PROMPT_MQ=2` | 1 | prompt attention in hi + lo halves, bit for bit the old output |
| `STRATA_ROUTER_OLD=1`, `STRATA_ROUTE_PER_TOKEN=1`, `STRATA_IQ_MMVQ_LDS=0` | | the older kernels, for A/B |
| `STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1`, `STRATA_PREGATE_STATS=1` | | per-request timing, GPU stage profile, prefetch score |

## Releases

Tag `vX.Y.Z` and push the tag. `.github/workflows/release.yml` builds and packages, then attaches the tarball, its
checksum and `install.sh` to a draft release. Write the notes using the structure of the earlier releases, check the
draft, then publish it.
