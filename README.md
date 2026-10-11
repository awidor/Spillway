# Spillway

Runs Qwen3.6-35B-A3B with 128K context on a laptop with an RTX 2070 Max-Q (8 GB VRAM) and 16 GB of RAM. On that
laptop it writes 69-76 tokens per second.

The model is 13.6 GB, so it can't fit on the GPU. Spillway keeps the experts the model uses most in VRAM and leaves the
rest in RAM, where the CPU or a PCIe copy handles them. The engine started as a cut-down copy of
[Strata](https://github.com/Niko1221/Strata), and the installer tunes it for the machine it runs on.

## Install

On Linux, with an NVIDIA driver 525 or newer:

```bash
curl -fsSL https://github.com/awidor/Spillway/releases/latest/download/install.sh | bash
```

It downloads the model if you don't have it yet, tries a handful of settings to find the fastest, and starts a service
on port 8095 that the rest of your network can reach. To keep it to the laptop itself, end the command with
`bash -s -- --local` instead, or add `--api-key <key>` if you want clients to need a key.

## Use

Open http://<laptop>:8095 for the dashboard. Point any OpenAI client at `http://<laptop>:8095/v1` with the model
`qwen3.6-35b-a3b`, or run `spillway chat` in a terminal.

`spillway status`, `spillway update` and `spillway uninstall` do what they say. `spillway help` lists the rest.

## Build

You need Docker. Run `./dev.sh setup && ./dev.sh build && ./dev.sh package`, and the release ends up in `out/release/`.
More in [docs/DEVELOPING.md](docs/DEVELOPING.md), and every benchmark is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## License

MIT. The engine keeps Strata's MIT notice; [engine/SOURCE.md](engine/SOURCE.md) lists what changed. The model is
byteshape's [Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF).
