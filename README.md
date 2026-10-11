# Spillway

Qwen3.6-35B-A3B with 128K context on an RTX 2070 Max-Q laptop: 8 GB VRAM, 16 GB RAM, Linux. It serves an OpenAI-compatible
API and a live dashboard.

The 13.6 GB model doesn't fit in 8 GB of VRAM. Spillway keeps the most-used experts in VRAM. The rest stay in pinned RAM,
where the CPU computes them or the GPU copies them over PCIe, with a prefetch one layer ahead. The model's own MTP layer
drafts the next tokens. The engine is a trimmed copy of [Strata](https://github.com/Niko1221/Strata), tuned for this one
machine.

## Speed

| Machine | Spillway | llama.cpp `--moe-cache-mib` | llama.cpp `--n-cpu-moe` |
|---|---:|---:|---:|
| RTX 2070 Max-Q laptop, short requests (v0.1.0) | 69-76 tok/s | | |
| RTX 2070 SUPER test server, 6K / 30K prompt (v0.4.0) | 81 / 74 tok/s | | |
| RTX 3080 test PC, short requests | 143-167 tok/s | 73-89 tok/s | 33-37 tok/s |

Every measurement and how it was taken: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Requirements

| | |
|---|---|
| GPU | RTX 2070 Max-Q (Turing, 8 GB); NVIDIA driver 525 or newer |
| RAM | 16 GB |
| OS | Linux x86_64 with systemd (not WSL), Python 3.10+ |
| Disk | 14 GB for the model, 1 GB for Spillway |

The installer refuses any GPU that isn't a Turing card with 8 GB. The build targets only sm_75.

## Install

```bash
curl -fsSL https://github.com/awidor/Spillway/releases/latest/download/install.sh | bash
```

The installer:
- checks the machine and downloads the model, or reuses it from the Hugging Face cache or `--model PATH`
- installs to `/opt/spillway` and adds the `spillway` command
- measures about 15 settings on the machine, then keeps the fastest
- starts `spillway.service` on `127.0.0.1:8095`, then checks a short request and a ~117K-token one

It sets the CPU governor to `performance` and turns on GPU persistence mode each time the service starts. `spillway uninstall`
puts both back.

| Option | |
|---|---|
| `--model PATH` | use this GGUF (byteshape's `Qwen3.6-35B-A3B-IQ3_S-3.06bpw.gguf`) |
| `--port N` | listen on port N (default 8095) |
| `--lan` | listen on the network, with a generated API key |
| `--local` | back to this machine only |

To pass options to the one-liner: `curl -fsSL .../install.sh | bash -s -- --lan`.

## Use

| | |
|---|---|
| Dashboard | `http://127.0.0.1:8095/` |
| OpenAI API | `http://127.0.0.1:8095/v1`, model `qwen3.6-35b-a3b` |
| Anthropic API | `http://127.0.0.1:8095/v1/messages` |
| Terminal chat | `spillway chat` |

```bash
curl http://127.0.0.1:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "qwen3.6-35b-a3b", "messages": [{"role": "user", "content": "Hello"}]}'
```

With `--lan`, send the key as `Authorization: Bearer <key>`. The dashboard takes it as `http://<host>:8095/#key=<key>`.
The installer prints both.

## Manage

| Command | |
|---|---|
| `spillway status` | service, API health, version, tuned settings |
| `spillway logs` | follow the service log |
| `spillway retune` | measure the settings again |
| `spillway report` | machine and results as Markdown, for an issue |
| `spillway update` | install the latest release |
| `spillway uninstall` | remove everything except the model |

## Build from source

```bash
./dev.sh setup && ./dev.sh build && ./dev.sh package
```

This needs Docker. The result is in `out/release/`. Tests, benchmarks and the repo layout: [docs/DEVELOPING.md](docs/DEVELOPING.md).

## License

MIT (`LICENSE`). `engine/` is Strata's code under its own MIT notice (`engine/LICENSE`), changed as listed in
[engine/SOURCE.md](engine/SOURCE.md). The model is byteshape's
[Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF), used as published.
