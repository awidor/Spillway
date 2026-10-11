# Changelog

All measurements: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## v0.5.1

- Dashboard: a token trace next to the expert heatmap. It shows the 8 experts each layer picked for the token being
  written and where each one ran (VRAM, prefetched over PCIe, copied over PCIe, CPU), plus the last 20 tokens' misses
  per layer. The latest picks are outlined on the heatmap, which now shows experts in ID order.

## v0.5.0

- One-line install: `curl -fsSL https://github.com/awidor/Spillway/releases/latest/download/install.sh | bash`. It checks
  the machine, reuses a downloaded model (Hugging Face cache or `--model`), and has `--port`, `--lan` (with an API key)
  and `--local`.
- The `spillway` command: `status`, `logs`, `chat`, `retune`, `report`, `update`, `uninstall`.
- The CPU governor and GPU persistence mode are set on every service start. Before, a reboot reset them, and the
  service then ran slower than the settings were tuned for.
- The installer stops with the engine's error when no setting starts; before, it installed a service that could not
  run.
- The page at `http://127.0.0.1:8095/` is a dashboard. It has a heatmap of all 10,240 experts (40 layers x 256),
  showing which are in VRAM and how often each is routed, read from the engine's new `EXPERTS` command. It also shows,
  per request, where the experts ran (VRAM, PCIe, CPU) next to decode speed, plus the VRAM split, decode speed by
  context, the last minute of GPU, CPU and PCIe use, and the installer's tuning runs.
- Release asset: `spillway-linux-x86_64.tar.gz` with a `.sha256`. The service is `spillway.service` in `/opt/spillway`.
  An install of v0.4.0 or older (/opt/strata) is replaced.
- Builds target only sm_75 (the RTX 2070 Max-Q).

## v0.4.0

- GPU prefetch: layer l's hidden state, run through layer l+1's router, predicts its top 8 experts. Up to 3 that miss
  VRAM are copied over PCIe during layer l.
- The PCIe copy runs on its own high-priority stream, beside the cached-expert kernels.
- Routing in one launch per window, the fast top-k kernel, and IQ3_XXS dense projections reading activations from shared
  memory (all bit-identical).
- The installer also tunes the PCIe share, the prefetch depth and the copy kernel's blocks.
- RTX 2070 SUPER test server: decode +40% at a 6K prompt (58 → 82 tok/s), +42% at 30K (52 → 74 tok/s).

## v0.3.0

- Prompt attention takes several prompt tokens per GPU block on tensor cores. The same kernel runs the decode windows.
- FP16 attention by default, llama.cpp's precision (next-token KL 1e-4 to 5e-4); `STRATA_PROMPT_MQ=2` restores hi + lo.
- RTX 2070 SUPER test server: a 30K prompt reads in 15.1 s instead of 26.3 s (1.74x); decode +7-8%.

## v0.2.0

- 128K context: `--max-context 131072 --kv-grow`. The K/V takes VRAM only as far as a request reaches.
- Flash-decoding attention: one K/V read per verify window, one partial per SM. 3-4x faster per layer; on the RTX 3080
  test PC, decode at a 30K prompt went from 106 to 135 tok/s.
- The installer checks a ~117K-token request.

## v0.1.0

- First release: the Strata copy with byteshape's fused gate/up experts, IQ2_XXS down kernels, expert layout 8, and BF16
  MMVQ for the model's MTP layer, plus the installer.
- Fix: long prompts crashed on Turing (null selection ids in #1821's int8 prompt attention).
- On the laptop: 69-76 tok/s on short requests.
