# Methodology

## Arms

| Arm | Definition |
|-----|------------|
| **base** | Same `llama-bench` / `llama-server` binary and Laguna Q4_K_M GGUF. All major custom SYCL fuse kill-switches set to `1`. |
| **tip** | Same binary/GGUF. Quality-safe tip: only `MUL_MAT_ADD`, `MOE_DUAL_DOWN`, and `MOE_DUAL_MULTITOKEN` disabled; remaining tip fuses enabled. |

## Formal serial

- Window: **pp512** (`-n 0`) and **tg128** (`-p 0`), **5 reps**
- Flags: `-ngl 99 -t 16 -ub 2048 -b 2048 -ctk f16 -ctv f16 -fa on`
- Device: `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, `ZE_AFFINITY_MASK=0`, `GGML_SYCL_DISABLE_GRAPH=1`
- Composite score vs historical pin: `decode_speedup^0.75 * prefill_speedup^0.25`

## Prefill ladder

Separate `llama-bench` runs for multi-prompt prefill (512…8192) and tg128.

## Long context

Server `c=32768`, flash-attn on, f16 KV. Needle-in-haystack (100 paragraphs, 3 depths) + Broadway dossier multi-QA (14 questions). Same prompts both arms.

## Single-agent speed

10 chat completions (5 prompts × 2), `max_tokens=96`, temp 0, seed 42. Report tg p50 and content sanity.

## Held-out tool pack

`ho-pack-v1.1` via OpenAI-compatible tool loop against mock tools. Score % of points.

## Agent Bench 69

`tool-eval-bench` 2.1.0. Context limited to 32k on this hardware for Laguna Q4; tool schemas often exceed that → automatic fails. Tip arm hit server abort under load — not banked as a tip score.

## Isolation of broken fuses

Enable-from-clean PPL on a short English corpus: `MUL_MAT_ADD` alone → PPL ~1e6; `MOE_DUAL_DOWN` → PPL failure; combination of the three kills restores PPL ~1.0 while keeping other fuses.
