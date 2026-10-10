# Spillway

[Strata](https://github.com/Niko1221/Strata)'s hybrid GPU + CPU mixture-of-experts engine, cut down and tuned for one
model on one machine:

- **Model:** Qwen3.6-35B-A3B, byteshape's 3.06 bpw GGUF with its MTP draft layer
  ([byteshape/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF)), used as published.
- **Box:** a headless Linux machine with an RTX 2070 Max-Q (8 GB, Turing), an i7-9750H (6 cores), 16 GB of DDR4 and
  PCIe 3.0.
- **Context:** at least 128K tokens.

The 13.6 GB model does not fit in 8 GB of VRAM. The engine keeps every expert in pinned RAM and the most used ones in a
VRAM cache, computes the misses on the CPU in place (or copies them over PCIe), and drafts with the model's own MTP
layer.

## Status

On the box (first report): 69-76 tok/s output on short requests, 2-3.6x the Gemma service it replaces. The rest was
measured on a stand-in (RTX 3080 10 GB in WSL2, Ryzen 7 5700X):

| On the stand-in, 3.06 bpw file | Decode tok/s |
|---|---:|
| llama.cpp, `--n-cpu-moe` | 33-37 |
| llama.cpp, `--moe-cache-mib` | 73-89 |
| Spillway, MTP off | 93-114 |
| **Spillway, MTP on (draft length 3)** | **143-167** |
| Spillway at the box's budget (VRAM held to 7.8 GB free, Turing code paths, 6 cores), MTP on | 143 |

Every number, with how it was measured: [RESULTS.md](RESULTS.md). The plan, the hardware facts behind each choice and
the speed projections: [PLAN.md](PLAN.md). What changed against upstream: [engine/SOURCE.md](engine/SOURCE.md).

## Layout

| Path | What |
|---|---|
| `engine/` | the engine: a copy of Strata `650cf81` (MIT) with the changes listed in `engine/SOURCE.md` |
| `kit/` | the build image and the scripts that go to the box: `s1.sh` (scouting / llama.cpp baseline), `s2.sh` (install, tune, run) |
| `perfmodel/` | the speed model, the hardware probe, and the dev-PC benchmark and parity scripts |
| `PLAN.md`, `RESULTS.md` | the plan and the measurements |

## Build

Everything is built in a container (Ubuntu 20.04, CUDA 12.8, sm_75 + sm_86), so the binaries need only glibc 2.31+
and an NVIDIA driver 525+ on the box. From the repository root:

```bash
docker build -t strata-build kit
git clone https://github.com/ggml-org/llama.cpp engine/third_party/llama.cpp
git -C engine/third_party/llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
docker run --rm -v "$PWD:/src" -v strata-work:/work strata-build bash /src/kit/build.sh
docker run --rm -v "$PWD:/src" -v strata-work:/work strata-build bash /src/kit/build_engine.sh
docker run --rm -v "$PWD:/src" strata-build bash /src/kit/package_s2.sh
```

That gives `out/s1kit/` (llama.cpp's `llama-batched-bench` plus the `scout` micro-benchmarks) and `out/s2kit/` (the
engine, the server and the installer). The llama.cpp checkout provides `gguf-py` for the packer.

## Run on the box

Copy `out/s2kit/` to the box and run:

```bash
bash s2kit/s2.sh
```

It downloads the model if it is not next to the kit, installs to `/opt/strata`, packs the model, tries nine settings
(draft length, VRAM reserve, CPU workers, expert kernel layout), starts `strata.service` with an OpenAI-compatible
API on `127.0.0.1:8095` and checks it. `bash s2kit/s2.sh --undo` removes it again.

For llama.cpp's numbers on the same box, copy `out/s1kit/` and run `bash s1kit/s1.sh`.

## License

MIT. `engine/` is Strata's code under its own MIT notice (`engine/LICENSE`), with the changes listed in
`engine/SOURCE.md`; the rest is under `LICENSE`.
