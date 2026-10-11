# Source

A copy of [Strata](https://github.com/Niko1221/Strata) (MIT, see `LICENSE`) at `650cf81`, the head of
[#1820](https://github.com/Niko1221/Strata/pull/1820), which contains upstream `main` @ `fb58e0d` and
[#1402](https://github.com/Niko1221/Strata/pull/1402) @ `48534f3` (the Qwen3.6 / `qwen35moe` port). Copied 10 Oct 2026.
Not a fork: nothing here goes upstream. 

## Changes against 650cf81

| Change | Files |
|---|---|
| byteshape's fused `ffn_gate_up_exps` read in place (gate rows then up rows per expert, the blob's own order) | `tools/iq_pack.py`, `include/strata/kernels/cpu/expert_layout.hpp`, `src/core/expert_source.cpp` |
| IQ2_XXS down projections in the GPU expert kernels | `src/kernels/cuda/iq_kernels.cu`, `src/kernels/native_grouped_parity.cpp` |
| Expert layout mode 8 is the default on every card (was Volta only) | `src/kernels/cuda/iq_kernels.cu` |
| BF16 weights in native MMVQ, for byteshape's MTP layer (draft K/V, shared expert); the main model's dense path unchanged | `src/kernels/cuda/native_mmvq.cu`, `src/core/native_dense.cpp` |
| Missing `<cstdint>` include | `include/strata/core/vram_floor.hpp` |
| Merged: #1821 (Turing int8 K/V prompt attention), #1825 (Turing 4-lane prompt scorer) | `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/cuda/qsa_select.cu`, their parity tests, `CMakeLists.txt` |
| Fix: #1821's int8 prefetch read `ids` without the dense (null ids) check, an illegal address on Turing for long prompts; the parity test now checks dense against ids 0..w-1 bit for bit | `src/kernels/cuda/qsa_prompt_attn.cu`, `src/kernels/qsa_prompt_attn_parity.cpp` |
| Decode attention for a dense selection (qwen35moe, int8 K/V) as flash decoding: one K/V read per window, one partial per SM, a merge with its loads in flight; the batched kernel stays for sparse selections and other formats (`STRATA_ATTN_DENSE=0` turns it off) | `src/kernels/cuda/qsa_decode_attn.cu`, its header, `src/core/verify.cpp`, `src/core/mtp.cpp` |
| Prompt attention for a dense selection (int8 K/V, 8 query heads per KV head): several prompt positions per block, 16 real rows per MMA tile, q and p as single FP16 halves by default, the precision of llama.cpp's flash attention (`STRATA_PROMPT_MQ=2`: hi + lo halves, bit for bit v1's output); its split mode runs decode windows on tensor cores (`STRATA_ATTN_TC=0`: the CUDA-core flash decoding) | `src/kernels/cuda/qsa_prompt_attn.cu`, its header, `src/kernels/cuda/qsa_decode_attn.cu` |
| Verify window, PCIe share: the copy kernel on its own high-priority stream beside the VRAM hits (`STRATA_FETCH_FORK`, default 4 blocks); graphs instantiated with node priorities | `src/core/verify.cpp`, its header, `src/kernels/cuda/verify_kernels.cu` |
| GPU expert prefetch: layer l+1's top 8 predicted from layer l's residual (its MoE norm and router), up to 3 misses copied during layer l and joined to layer l+1's hits launch (`pf_merge`); the host plan gives prefetched misses to the GPU (`STRATA_PREFETCH`, `STRATA_PREFETCH_MAX`, `STRATA_PREGATE_STATS`) | `src/core/verify.cpp`, its header, `src/kernels/cuda/verify_kernels.cu`, its header, `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp`, `src/program/generate.cpp` |
| The window's routing in one launch on CUDA too (was gfx906 only; bitwise, `route_window_parity`) | `src/core/verify.cpp` |
| The fast router top-k on CUDA too (bitwise; `cuda_router_fast` runs the HIP test through a name shim; `STRATA_ROUTER_OLD=1`) | `src/kernels/cuda/router_top10.cu`, `CMakeLists.txt`, `tests/cuda_shim/` |
| i-quant dense MMVQ reads the activations from shared memory, 2 rows a warp (bitwise; `STRATA_IQ_MMVQ_LDS=0`) | `src/kernels/cuda/iq_kernels.cu` |
| Removed: SYCL engine, installer (`setup.py` and its tests), MCP server tool, Docker files, Windows launchers, translated READMEs | |
| Removed: upstream's docs and benchmark results, data for other models (other expert profiles and draft vocabularies, the Qwen3.8 control vector) | `docs/`, `bench/`, `data/` |
| The web page is Spillway's dashboard (`/metrics` as charts) instead of the chat app; `/metrics` adds the installer's tuning runs (`tuning_rows`, `tune.csv`) | `serve/web/`, `serve/server.py`, `serve/test_tuning.py` |
| `chat.py`: port 8095, `--api-key` | `chat.py` |
| Line endings normalized to LF | all text files |

Measurements for each change: `../docs/BENCHMARKS.md`.
