# Measured results

RTX 3080 test PC = the dev PC's RTX 3080 10 GB in WSL2 (Windows desktop on the same card), Ryzen 7 5700X, 3.06 bpw file,
greedy, 4K context, `dev/bench_engine.sh` (one coding prompt, 128 tokens) unless noted.

## llama.cpp (781dbc5), scouting kit dry run, RTX 3080 test PC

| Config | pp2048 | tg after 2K | pp8192 | tg after 8K |
|---|---:|---:|---:|---:|
| `--n-cpu-moe 15` (as many layers' experts on the GPU as fit) | 860 | 33.3 | 1135 | 36.7 |
| `-cmoe --moe-cache-mib 7367` | 584 | 72.8 | 956 | 89.2 |

## Engine copy (650cf81 + byteshape fixes), RTX 3080 test PC

| Date | Change | Decode tok/s | ms/window | Notes |
|---|---|---:|---:|---|
| 10 Oct | baseline, MTP off, `--spec 2` | 76.8 | 13.2 | profile on; cache 5,465 slots (5.9 GiB). GPU: VRAM-hit experts 5.1 ms/window (~63 GB/s effective) |
| 10 Oct | expert layout mode 8 default (was 0 on non-Volta) | 114.0 | — | profile on; VRAM hits 5.1 -> 1.7 ms/window. Sweep: mode 0 72.9, 2 86.0, 5 103.6, 6 113.3, 7 110.3, 8 114.0 (1, 4 did not run). Greedy output identical to mode 0 over 128 tokens (fixed cache 4000) |
| 10 Oct | BF16 MMVQ for byteshape's MTP layer (draft K/V + shared expert), MTP on | **148.4** | 18.2 | `--spec 3 --spec-min-p 0.5`, `data/draft_vocab.bin`; 2.69 tokens/window, drafts 162/184 accepted. spec 2: 137.7, spec 4: 132.2; no MTP: 92.6 (256 tokens, profile off) |

## Parity, greedy vs llama.cpp `llama-completion` on the same GGUF, 64 tokens (`dev/parity_greedy.py`)

| Prompt | Engine, MTP off | Engine, MTP k=3 |
|---|---|---|
| code (merge sorted lists) | identical | identical |
| chat (stay focused) | split at 46: "specific" vs "certain" | identical |
| German (Kühlschrank) | split at 28: "um die Temperatur zu" vs "wodurch sich die Temperatur" | split at 29 |

The splits are near-tie wordings, the IQ3-level rounding upstream documents for this model (docs/QWEN36.md).

## The laptop's budget on the RTX 3080 test PC: VRAM held to 7,800 MiB free (`scout hold`), `STRATA_EMULATE_CC=75`, 6 cores (`taskset -c 0-11`, `--pool-workers 5`), MTP k=3

| Expert cache | Decode tok/s | ms/window | Note |
|---|---:|---:|---|
| auto | — | — | refuses: the WDDM-only post-touch haircut (`under_wddm()`, WSL counts) reads 0 MiB free; native Linux skips it |
| 6,118 slots (auto's pre-touch size, 700 MiB reserve) | 55.0 | 48.5 | WDDM spilled VRAM to system RAM (GPU wait 40 ms): test PC artifact |
| 5,400 slots (53% of the 10,240 experts) | **142.8** | 18.9 | 2.69 tokens/window; GPU wait 10.9, CPU pool 4.6, draft 1.3 ms/window |

The real slot count and the reserve are settled on the laptop (the installer's tuning: `--vram-reserve-mib`).

## Turing pull requests

| PR | Taken? | Why | Check |
|---|---|---|---|
| #1821 int8 K/V prompt attention (Turing) | yes, 3-way merge | self-contained prompt kernel | `qsa_prompt_attn_parity` PASS, also with `STRATA_EMULATE_CC=75` |
| #1825 4-lane prompt scorer (Turing) | yes, 3-way merge | self-contained | `qsa_scores_parity` bitwise identical (native and EMULATE_CC=75) |
| #1437 sm_75 interleaved verify projections | no | its table holds only Flash-Next shapes (2560/6144 wide); Qwen3.6's 2048-wide projections fall back, so no effect; 650cf81 already has the per-arch IL tables | — |
| #1581 SM75 prompt / VRAM loans | no | predates #1820 and restructures the MTP expert load for experts.bin; 12 conflicts with #1820's qwen35moe in-file draft layer; prompt-only gains | — |

After the merges: `native_grouped_parity` 0 failures (IQ3_XXS/IQ2_XXS bitwise), `mmvq_multi_parity` ok; decode 160.3 / 142.8 tok/s in two runs (MTP k=3, run-to-run spread on the WSL test PC ~±6%).

## Installer dry run on the RTX 3080 test PC (10 Oct)

Install + pack 10 s; tuning 9 engine starts (~15 s each); service up and answering through the OpenAI API.

| Candidate | tok/s |
|---|---:|
| spec 2 / 3 / 4 | 155.8 / **167.1** / 159.1 |
| spec 3, VRAM reserve 450 / 250 MiB | 145.5 / 131.1 (bigger cache spills under WDDM on the test PC; the laptop decides) |
| spec 3, 5 / 4 pool workers | 166.9 / 161.6 |
| spec 3, expert layout 6 / 7 | 133.7 / 155.4 |

Through the service: 160.8 tok/s decode, drafts 227/270. `--undo` removed the service and /opt/strata.

## On the laptop: first report (10 Oct)

RTX 2070 Max-Q 8 GB, i7-9750H, the service with the installer's tuned settings, against the Gemma service it replaces.
Output tokens/s per request, one run each, each model counted with its own tokenizer.

| Test | Gemma | Spillway |
|---|---:|---:|
| Field extraction | 35.1 | **68.9** |
| Factual text | 21.0 | **75.5** |
| Longer prompt | ok | GPU illegal memory access (fixed below) |

Check: the RTX 3080 test PC at the laptop's budget gave 142.8 tok/s. Decode there is bound by reading cached experts from VRAM,
and the 2070 Max-Q has 384 GB/s against the 3080's 760 (0.51x): 142.8 x 0.51 = 72 tok/s expected, 69-76 measured.
Inside docs/design/plan.md's projection (63 / 115 / 173 lo / mid / hi with MTP), near lo.

## Fix: long prompts crashed on Turing (laptop report, 10 Oct)

On the laptop, a longer prompt failed with a GPU illegal memory access. It reproduces on the RTX 3080 test PC with
`STRATA_EMULATE_CC=75` and the installer's flags (`--max-context 32768 --kv int8 --mtp`, cache 5,610 slots).
`CUDA_LAUNCH_BLOCKING=1` traced it to `qsa_prompt_attn_batch`: #1821's int8 prefetch read `ids[...]` with no null
check, but prefill passes null ids for dense attention. sm_80+ takes the cp.async kernel, which has the check, so the
crash only shows on Turing. Fix: dense cell = index, as in the other kernels. `qsa_prompt_attn_parity` now also
checks null ids against ids 0..w-1 bit for bit; it crashed under EMULATE_CC=75 before the fix and passes after (int8
and fp16, ctx 8K and 32K).

| Prompt, RTX 3080 test PC, MTP k=3 | Before | After: prompt tok/s | After: decode tok/s (64 tokens) |
|---|---|---:|---:|
| 6,019 tokens, EMULATE_CC=75 | illegal memory access | 2,498 | 112.2 |
| 30,019 tokens, EMULATE_CC=75 | — | 1,779 | 29.8 |
| 30,019 tokens, native sm_86 | — | 1,858 | 66.6 |

## Long context (128K target, 10 Oct)

The target moved to 128K tokens. qwen35moe attends every cell, and the batched decode-attention kernel read the whole
K/V once per window token, then merged one partial per 64 cells serially. New: flash decoding for the dense
selection (int8 K/V): a block per (SM-sized split, KV head, up to 4 window tokens) stages each 32-cell tile in shared
memory once, keeps an online softmax per row, and the merge folds one partial per split with its loads in flight.
Each token's result is the same alone as in its window (bit for bit); against the batched kernel the largest
difference is 3e-9 (`dev/attn_bench.cu`, which exits non-zero otherwise).

| Attention per layer, RTX 3080 test PC, int8 K/V | Batched kernel | Flash decoding |
|---|---:|---:|
| 6K cells, 1 / 4 tokens | 121 / 261 us | 41 / 71 us |
| 30K cells, 1 / 4 tokens | 424 / 1,050 us | 122 / 256 us |
| 128K cells, 1 / 4 tokens | 1,739 / 4,070 us | 425 / 919 us |

End to end, `STRATA_EMULATE_CC=75`, MTP k=3, `--kv int8`, 128 tokens (`STRATA_ATTN_DENSE=0` = the batched kernel):

| Prompt | Batched | Flash decoding |
|---|---|---|
| 6K tokens | 13.9 ms/window, 156 tok/s | 13.1 ms/window, 157 tok/s |
| 30K tokens | 19.5 ms/window, 106 tok/s | 14.4 ms/window, **135 tok/s** |
| 120K tokens, `--max-context 131072 --kv-grow` | not run | 22.1 ms/window, **89 tok/s**; prompt read 149 s (805 tok/s) |

`--kv-grow` at `--max-context 131072`: the expert cache starts at 5,464 slots (5.86 GiB) instead of 4,335 (4.65 GiB)
with the whole 128K K/V allocated (1.38 GB int8), and gives slots back only as a request's context grows. The installer now
runs `--max-context 131072 --kv int8 --kv-grow` and checks one ~117K-token request through the API.

Left at 128K: decode attention is compute-bound on CUDA cores at 4 tokens (145 GB/s of K/V); tensor cores are the
next step. Prompt reading at 120K is 805 tok/s on the RTX 3080 test PC, so about 2.5 minutes there and more on the laptop.

## Dev server (11 Oct): a real Turing card

From here the test machine is a server with a desktop **RTX 2070 SUPER** (TU104, sm_75, 40 SMs, 448 GB/s, 8 GB) and
an i3-6100 (2 cores / 4 threads, far weaker than the laptop's 6-core i7). Builds and runs happen in a container there
(`dev/devbox.sh`), capped at 16 GB of RAM like the laptop. GPU kernels compare directly with the laptop's 2070 Max-Q
(36 SMs, 384 GB/s); whole-engine decode speeds do not, because the i3 computes the CPU's share of the experts.

## Prompt attention: several positions per block (dense, int8 K/V, 8 query heads per KV head)

The Turing prompt-attention kernel (v1) ran one block per prompt position: its 16-row MMA tile held one position's 8
query heads and 8 zero rows, and every position gathered and converted the whole K/V. New: a block takes 2 positions
(Turing) or 4 (sm_80+), 16 real rows per tile, each K/V chunk gathered once for all of them. Per-row arithmetic is
v1's (q and p as FP16 hi + lo halves, FP32-level): the same maximum error against the FP32 decode kernel (3e-6
relative), and the same greedy tokens as v1 over 24 tokens of a 6K prompt (a repeat run of v1 itself splits at
token 4: the engine is not bit-reproducible run to run). `dev/prompt_bench.cu`, 2,048 positions at 30K:

| RTX 2070 SUPER | Time | |
|---|---:|---|
| v1 | 213 ms | |
| **2 positions per block (Turing default)** | **109 ms** | 1.96x |
| 4 per block | 122 ms | 48 KB of shared memory: one block per Turing SM |
| 2 / 4 per block, FP16 q and p (no lo half, `STRATA_PROMPT_MQ=2f` / `4f`) | 80 / 63 ms | 2.7x / 3.4x; FP16-level error (5.8e-4 relative), as llama.cpp's flash attention |

With each position's V scale taken over the cells it attends (as v1 effectively does), the new kernel's output is
**bit for bit v1's** (same output hash at 30K x 2,048, 6K x 513, 1.5K x 1.5K and 100 x 37 positions).

Whole prompt, 30K tokens, `--max-context 131072 --kv int8 --kv-grow`: prompt attention 16.3 -> 8.3 s, prompt read
26.4 -> 18.4 s (1,136 -> 1,628 tok/s). At 6K: 3.1 -> 2.7 s.

## Decode attention on tensor cores (same kernel, split mode)

The prompt kernel's tile (2 positions x 8 heads = 16 rows) fits a verify window, so the same kernel with the cells
split across the GPU (block = split, KV head, 2 positions; partials into the flash-decoding merge) replaces the
CUDA-core flash decoding on sm_75+ (`STRATA_ATTN_TC=0` turns it off). FP32-level (hi + lo), and a token's result is
the same alone as in its window, bit for bit. `dev/attn_bench.cu`, RTX 2070 SUPER, per layer:

| Cells, window tokens | Batched (original) | CUDA-core flash decoding | Tensor cores |
|---|---:|---:|---:|
| 6K, 4 | 474 us | 136 us | **84 us** |
| 30K, 3-4 | 1,921-2,059 us | 484 us | **299 us** |
| 128K, 4 | 8,711 us | 2,048 us | **1,024 us** |

End to end at 30K (dev server, MTP k=3): QSA attention 3.9 -> 2.0 ms per window, decode 48.9 -> 52.1 tok/s. The i3's
CPU expert work is 19.5 of the 38 ms window here; on the laptop's 6-core i7 it is a fraction of that, so the GPU-side
gain is a larger share there.

## FP16 attention as the default (11 Oct)

Both new kernels can take q and the probabilities as single FP16 halves instead of hi + lo pairs: half the MMAs,
the precision of llama.cpp's CUDA flash attention, which is how byteshape's file normally runs. Next-token
distribution right after the prompt (`STRATA_DUMP_FIRST_LOGITS`, `dev/logit_compare.py`), against the hi + lo
arithmetic:

| Compared with hi + lo attention, int8 K/V | KL, 6K prompt | KL, 30K prompt | Top token / top 5 |
|---|---:|---:|---|
| the same run again | 0 | 0 | same / 5 of 5 |
| **FP16 attention (now the default)** | **5.2e-4** | **1.0e-4** | same / 5 of 5 |
| an FP16 K/V cache instead of int8 (for scale) | 2.1e-3 | 1.4e-3 | same / 5 of 5 |

FP16 attention moves the output a quarter to a tenth as much as the int8 cache already does (upstream ships its
CPU prompt share at a mean KL of about 0.004). Defaults now: prompts 4 positions per block, decode 2 per block for
windows of 1-2 tokens and 4 above; `STRATA_PROMPT_MQ=2` / `STRATA_ATTN_TC=2` give back the hi + lo arithmetic.
Window independence still holds (a token alone = in its window, 1-5 tokens).

| RTX 2070 SUPER | v0.2.0 | hi + lo | **FP16 (default)** |
|---|---:|---:|---:|
| Prompt attention, 30K x 2,048 positions | 213 ms | 109 ms | **72 ms** |
| Whole 30K prompt | 26.4 s | 18.4 s | **15.1 s** |
| Decode attention per layer, 128K, 1 / 4 tokens | 849 / 2,048 us | 490 / 1,012 us | **414 / 651 us** |
| Decode attention per layer, 6K, 4 tokens | 136 us | 84 us | **77 us** |

## v0.3.0 against v0.2.0 (11 Oct), the packaged kit on the dev server

Same binary; v0.2.0's kernels switched back on with `STRATA_PROMPT_MQ=0 STRATA_ATTN_TC=0`. RTX 2070 SUPER, i3-6100,
`--max-context 131072 --kv int8 --kv-grow`, MTP k=3, 128 tokens.

| | v0.2.0 kernels | v0.3.0 | |
|---|---:|---:|---|
| Prompt read, 6K tokens | 3.11 s | 2.65 s | 1.17x |
| Prompt read, 30K tokens | 26.3 s | 15.1 s | 1.74x |
| Decode, 6K prompt (three runs each) | 55.5-57.4 tok/s (one outlier 47.4) | 59.2-60.1 tok/s | +7% |
| Decode, 30K prompt (one run each) | 50.0 tok/s | 54.2 tok/s | +8% |

The i3's CPU expert work is about half of every decode window here; the laptop's 6-core i7 takes a fraction of that, so
the attention savings are a larger share of its windows.

## Decode: the experts that miss the VRAM cache (11 Oct, dev server)

A GPU-bound decode window (`--pcie-frac 1`, 6K prompt) spent 7.6 of 27 ms waiting for missed experts to cross PCIe
(about 80 MB a window). The link itself gives 11.8 GB/s to both the copy kernel and the copy engine
(`dev/zc_bench.cu`), so the work was in when the copies run, not how. Each step, the laptop's flags (128K, int8 K/V,
`--kv-grow`, MTP k=3), 128 tokens, A/B by its switch, generated tokens identical unless noted:

| Step | Before | After | Switch |
|---|---:|---:|---|
| The PCIe copy on its own stream, beside the VRAM hits' kernels instead of after them (4 blocks) | 82.1 | 89.0 tok/s | `STRATA_FETCH_FORK=0` |
| The window's tokens routed in one launch (the 256-expert router had one launch per token on CUDA; bitwise, `route_window_parity`) | 84.8 | 88.7 tok/s | `STRATA_ROUTE_PER_TOKEN=1` |
| Side streams at the highest priority (`cudaGraphInstantiateFlagUseNodePriority`): their few blocks no longer wait behind whole grids | 89.0 | 90.2 tok/s | - |
| GPU prefetch: layer l's residual through layer l+1's MoE norm and router predicts its top 8; up to 3 of them that miss VRAM are copied during layer l and joined to layer l+1's hits launch | 90.4 | 99.5 tok/s | `STRATA_PREFETCH=0`, `STRATA_PREFETCH_MAX` |
| The fast top-k kernel (AMD's, bitwise; `cuda_router_fast`): 17.5 -> 9.2 us a call | 99.3 | 101.0 tok/s | `STRATA_ROUTER_OLD=1` |
| IQ3_XXS dense GEMVs read the activations from shared memory (bitwise; qkv at T=3 79 -> 44 us, `dev/iq_mmvq_bench.cu`) | 101.0 | 104.4 tok/s | `STRATA_IQ_MMVQ_LDS=0` |

The prediction (`STRATA_PREGATE_STATS=1` scores it): top-8 a layer early names 85% of the routed experts and 71% of
the misses (top-12: 87%, top-16: 93%, at 2-4x the copies). Wider prefetches lose: the link is shared with the
window's own copies, its flags and the CPU's rows (top-16: 54 tok/s). The host's plan reads which experts were
prefetched and gives them to the GPU whatever `--pcie-frac` says, so the CPU only computes unpredicted misses: at
`--pcie-frac 0.33` decode went 60.3 -> 79.4 tok/s (tokens differ there, as between any two CPU/GPU splits).
Correctness never depends on the host's table: the GPU joins a blob to the hits only when its own prefetch holds it.

Tried and dropped: the host-side look-ahead already in the engine (`STRATA_FS_SLOTS`, CPU router + DMA): no gain, its
copies land after the next layer is planned; 4 loads in flight a thread in the copy kernel (91 tok/s, the link's
small control reads queue behind it); 4 or 8 rows a block in native MMVQ (no gain); shared-memory activations in
native MMVQ (IQ4_XS 15-20% faster, Q5_K slower at T >= 3).

## v0.4.0 against v0.3.0 (11 Oct), dev server

The v0.3.0 kit's engine against this build, same flags as the laptop (`--max-context 131072 --kv int8 --kv-grow`,
MTP k=3), 128 tokens, `dev/release_ab.sh`:

| | v0.3.0 | v0.4.0 | |
|---|---:|---:|---|
| Decode, 6K prompt, default PCIe share (two runs) | 58.1-58.4 tok/s | 81.1-82.3 tok/s | +40% |
| Decode, 30K prompt, default PCIe share (two runs) | 51.8-52.9 tok/s | 73.6-75.0 tok/s | +42% |
| Decode, 6K prompt, `--pcie-frac 1` | 77.2 tok/s | 104.7 tok/s | +36%, same tokens |
| Decode, 30K prompt, `--pcie-frac 1` | 65.9 tok/s | 86.0 tok/s | +31%, same tokens |
| Prompt read, 6K / 30K | 2.61 / 15.1 s | 2.60 / 15.1 s | unchanged |

The installer now also tries `--pcie-frac 0.66` and `1`, the prefetch off and at 2 or 5 a layer, and 8 copy blocks, so the laptop
picks its own CPU/PCIe split: the i3 here is far weaker than the laptop's i7.

## Selected Strata v0.1.43.1 backports, RTX 2070 Max-Q

RTX 2070 Max-Q 8 GB, i7-9750H, 16 GB RAM, CUDA 12.4, GCC 13, sm_75. Qwen3.6-35B-A3B, the same 3.06 bpw GGUF and pack; 131072 context, int8 KV, kv-grow, MTP k=3, reserve 450 MiB, four CPU workers.

Two alternating runs per mode and prompt length, a synthetic Python coding task, 1024 output tokens per run. Baseline and overlap used `--pcie-frac 0.35`; only the measured controller could refine that split.

| Median decode tok/s | 6K input | 30K input |
|---|---:|---:|
| v0.5.2 baseline | 91.85 | 83.55 |
| Backports, both options off | 92.95 | 83.65 |
| Adaptive overlap only | 96.65 | 85.25 |
| Measured CPU/PCIe split only | 98.35 (+7.08%) | 84.80 (+1.50%) |

The measured split's two runs beat both baseline runs at each prompt length. Overlap had more spread (6K: 100.4 and 92.9 tok/s); a separate timing diagnostic exercised 110 overlapped windows. The options remain off by default. No combined-options benchmark was run.

The unchanged baseline's repeated full-output hashes differed at both lengths, so byte-identical full replies are not claimed. This small synthetic comparison is not a broad quality or sustained-load result. Four GPU parity programs passed: `qsa_prompt_attn_parity`, `native_grouped_parity`, `route_window_parity`, `cuda_router_fast`. The parser, structured-output, server and reasoning-tool suites passed 308 tests. A synthetic 100186-token retrieval with beginning/middle/end markers returned correct JSON in 133.809 s; a client disconnect released the single slot in 0.253 s.
