# Measured results

Proxy = the dev PC's RTX 3080 10 GB in WSL2 (Windows desktop on the same card), Ryzen 7 5700X, 3.06 bpw file,
greedy, 4K context, `perfmodel/bench_engine.sh` (one coding prompt, 128 tokens) unless noted.

## llama.cpp (781dbc5), `out/s1kit` dry run, 3080 proxy

| Config | pp2048 | tg after 2K | pp8192 | tg after 8K |
|---|---:|---:|---:|---:|
| `--n-cpu-moe 15` (as many layers' experts on the GPU as fit) | 860 | 33.3 | 1135 | 36.7 |
| `-cmoe --moe-cache-mib 7367` | 584 | 72.8 | 956 | 89.2 |

## Engine copy (650cf81 + byteshape fixes), 3080 proxy

| Date | Change | Decode tok/s | ms/window | Notes |
|---|---|---:|---:|---|
| 10 Oct | baseline, MTP off, `--spec 2` | 76.8 | 13.2 | profile on; cache 5,465 slots (5.9 GiB). GPU: VRAM-hit experts 5.1 ms/window (~63 GB/s effective) |
| 10 Oct | expert layout mode 8 default (was 0 on non-Volta) | 114.0 | — | profile on; VRAM hits 5.1 -> 1.7 ms/window. Sweep: mode 0 72.9, 2 86.0, 5 103.6, 6 113.3, 7 110.3, 8 114.0 (1, 4 did not run). Greedy output identical to mode 0 over 128 tokens (fixed cache 4000) |
| 10 Oct | BF16 MMVQ for byteshape's MTP layer (draft K/V + shared expert), MTP on | **148.4** | 18.2 | `--spec 3 --spec-min-p 0.5`, `data/draft_vocab.bin`; 2.69 tokens/window, drafts 162/184 accepted. spec 2: 137.7, spec 4: 132.2; no MTP: 92.6 (256 tokens, profile off) |

## Parity (phase 1), greedy vs llama.cpp `llama-completion` on the same GGUF, 64 tokens (`perfmodel/parity_greedy.py`)

| Prompt | Engine, MTP off | Engine, MTP k=3 |
|---|---|---|
| code (merge sorted lists) | identical | identical |
| chat (stay focused) | split at 46: "specific" vs "certain" | identical |
| German (Kühlschrank) | split at 28: "um die Temperatur zu" vs "wodurch sich die Temperatur" | split at 29 |

The splits are near-tie wordings, the IQ3-level rounding upstream documents for this model (docs/QWEN36.md).

## Box budget on the proxy (phases 2-3): VRAM held to 7,800 MiB free (`scout hold`), `STRATA_EMULATE_CC=75`, 6 cores (`taskset -c 0-11`, `--pool-workers 5`), MTP k=3

| Expert cache | Decode tok/s | ms/window | Note |
|---|---:|---:|---|
| auto | — | — | refuses: the WDDM-only post-touch haircut (`under_wddm()`, WSL counts) reads 0 MiB free; native Linux skips it |
| 6,118 slots (auto's pre-touch size, 700 MiB reserve) | 55.0 | 48.5 | WDDM spilled VRAM to system RAM (GPU wait 40 ms): proxy artifact |
| 5,400 slots (53% of the 10,240 experts) | **142.8** | 18.9 | 2.69 tokens/window; GPU wait 10.9, CPU pool 4.6, draft 1.3 ms/window |

The real slot count and the reserve are settled on the box (S2 autotune: `--vram-reserve-mib`).

## Phase 4: Turing PRs

| PR | Taken? | Why | Check |
|---|---|---|---|
| #1821 int8 K/V prompt attention (Turing) | yes, 3-way merge | self-contained prompt kernel | `qsa_prompt_attn_parity` PASS, also with `STRATA_EMULATE_CC=75` |
| #1825 4-lane prompt scorer (Turing) | yes, 3-way merge | self-contained | `qsa_scores_parity` bitwise identical (native and EMULATE_CC=75) |
| #1437 sm_75 interleaved verify projections | no | its table holds only Flash-Next shapes (2560/6144 wide); Qwen3.6's 2048-wide projections fall back, so no effect; 650cf81 already has the per-arch IL tables | — |
| #1581 SM75 prompt / VRAM loans | no | predates #1820 and restructures the MTP expert load for experts.bin; 12 conflicts with #1820's qwen35moe in-file draft layer; prompt-only gains | — |

After the merges: `native_grouped_parity` 0 failures (IQ3_XXS/IQ2_XXS bitwise), `mmvq_multi_parity` ok; decode 160.3 / 142.8 tok/s in two runs (MTP k=3, run-to-run spread on the WSL proxy ~±6%).

## S2 kit dry run on the proxy (`bash s2kit/s2.sh`, 10 Oct)

Install + pack 10 s; tuning 9 engine starts (~15 s each); service up and answering through the OpenAI API.

| Candidate | tok/s |
|---|---:|
| spec 2 / 3 / 4 | 155.8 / **167.1** / 159.1 |
| spec 3, VRAM reserve 450 / 250 MiB | 145.5 / 131.1 (bigger cache spills under WDDM on the proxy; the box decides) |
| spec 3, 5 / 4 pool workers | 166.9 / 161.6 |
| spec 3, expert layout 6 / 7 | 133.7 / 155.4 |

Through the service: 160.8 tok/s decode, drafts 227/270. `--undo` removed the service and /opt/strata.

## On the box: first report (10 Oct)

RTX 2070 Max-Q 8 GB, i7-9750H, `strata.service` with S2's tuned settings, against the Gemma service it replaces.
Output tokens/s per request, one run each, each model counted with its own tokenizer.

| Test | Gemma | Spillway |
|---|---:|---:|
| Field extraction | 35.1 | **68.9** |
| Factual text | 21.0 | **75.5** |
| Longer prompt | ok | GPU illegal memory access (fixed below) |

Check: the proxy at the box's budget gave 142.8 tok/s. Decode there is bound by reading cached experts from VRAM,
and the 2070 Max-Q has 384 GB/s against the 3080's 760 (0.51x): 142.8 x 0.51 = 72 tok/s expected, 69-76 measured.
Inside PLAN.md's projection (63 / 115 / 173 lo / mid / hi with MTP), near lo.

## Fix: long prompts crashed on Turing (laptop report, 10 Oct)

On the box, a longer prompt failed with a GPU illegal memory access. It reproduces on the proxy with
`STRATA_EMULATE_CC=75` and the S2 flags (`--max-context 32768 --kv int8 --mtp`, cache 5,610 slots).
`CUDA_LAUNCH_BLOCKING=1` traced it to `qsa_prompt_attn_batch`: #1821's int8 prefetch read `ids[...]` with no null
check, but prefill passes null ids for dense attention. sm_80+ takes the cp.async kernel, which has the check, so the
crash only shows on Turing. Fix: dense cell = index, as in the other kernels. `qsa_prompt_attn_parity` now also
checks null ids against ids 0..w-1 bit for bit; it crashed under EMULATE_CC=75 before the fix and passes after (int8
and fp16, ctx 8K and 32K).

| Prompt, proxy, MTP k=3 | Before | After: prompt tok/s | After: decode tok/s (64 tokens) |
|---|---|---:|---:|
| 6,019 tokens, EMULATE_CC=75 | illegal memory access | 2,498 | 112.2 |
| 30,019 tokens, EMULATE_CC=75 | — | 1,779 | 29.8 |
| 30,019 tokens, native sm_86 | — | 1,858 | 66.6 |

## Phase 7: long context (128K target, 10 Oct)

The target moved to 128K tokens. qwen35moe attends every cell, and the batched decode-attention kernel read the whole
K/V once per window token, then merged one partial per 64 cells serially. New: flash decoding for the dense
selection (int8 K/V): a block per (SM-sized split, KV head, up to 4 window tokens) stages each 32-cell tile in shared
memory once, keeps an online softmax per row, and the merge folds one partial per split with its loads in flight.
Each token's result is the same alone as in its window (bit for bit); against the batched kernel the largest
difference is 3e-9 (`perfmodel/attn_bench.cu`, which exits non-zero otherwise).

| Attention per layer, proxy, int8 K/V | Batched kernel | Flash decoding |
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
with the whole 128K K/V allocated (1.38 GB int8), and gives slots back only as a request's context grows. S2 now
runs `--max-context 131072 --kv int8 --kv-grow` and checks one ~117K-token request through the API.

Left at 128K: decode attention is compute-bound on CUDA cores at 4 tokens (145 GB/s of K/V); tensor cores are the
next step. Prompt reading at 120K is 805 tok/s on the proxy, so about 2.5 minutes there and more on the box.
