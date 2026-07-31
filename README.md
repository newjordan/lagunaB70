# Laguna B70

**Faster text generation for Laguna-XS-2.1 on the Intel Arc Pro B70. 108 → 136 tok/s.**

Custom llama.cpp SYCL kernels, plus the measurements that back them up.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/graphs/benchmark-dark.svg">
    <img src="assets/graphs/benchmark-light.svg" width="880"
         alt="Writing an answer: stock 108 tok/s, lagunaB70 136 tok/s, 26 percent faster. Reading a prompt: 1142 to 1174 tok/s at 512 tokens, 1954 to 2029 at 2048, 1933 to 2002 at 4096, 1880 to 1950 at 8192.">
  </picture>
</p>

`stock` is the same binary and the same model file with these kernels switched off.
Both arms run one request at a time. Numbers come from one matched run —
[`results/AB_REPORT.md`](results/AB_REPORT.md), method in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

<details><summary>Same numbers as a table</summary>

| | stock | lagunaB70 | |
|---|---:|---:|---:|
| Writing an answer (128 tokens) | 108 tok/s | **136 tok/s** | +26% |
| Reading a 512-token prompt | 1142 tok/s | 1174 tok/s | +3% |
| Reading a 2048-token prompt | 1954 tok/s | 2029 tok/s | +4% |
| Reading a 4096-token prompt | 1933 tok/s | 2002 tok/s | +4% |
| Reading an 8192-token prompt | 1880 tok/s | 1950 tok/s | +4% |

</details>

## How it gets faster

Generating one token spends most of its time launching lots of tiny GPU jobs and
re-reading the same weights out of memory. These kernels glue those jobs together
so the data is touched once:

- gate + up + SwiGLU as a single expert kernel
- the MoE router (top-8 of 256 experts) as one kernel, replacing a five-kernel chain
- weighted MoE-down and packed reduce
- RMS-norm + multiply, RoPE + row-store, residual adds — fused into their neighbours
- flash-attention VEC GQA on the attention path

Writing speed is where this pays off. Prompt reading was already near the card's
memory-bandwidth limit, so it gains about 3–4%.

## Three kernels stay off

Perplexity isolation caught three fuses corrupting logprobs or multi-token
stability, so they ship disabled:

```bash
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
```

Turning all of them on reports a much larger prefill gain. That number is excluded
here because the model's output quality falls apart — see [`notes/`](notes).

## Run it

```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1

./llama-bench -m Laguna-XS-2.1-Q4_K_M.gguf \
  -ngl 99 -t 16 -ub 2048 -b 2048 -ctk f16 -ctv f16 -fa on -r 5 -p 512 -n 0
./llama-bench -m Laguna-XS-2.1-Q4_K_M.gguf \
  -ngl 99 -t 16 -ub 2048 -b 2048 -ctk f16 -ctv f16 -fa on -r 5 -p 0 -n 128
```

| | |
|--|--|
| Model | Laguna-XS-2.1 Q4_K_M (~19 GiB) |
| Device | Intel Arc Pro B70, Level-Zero |
| Engine | llama.cpp SYCL (`ggml-sycl`) |
| Window | pp512, tg128 · 5 reps · f16 KV · flash-attn on |

## Layout

```text
assets/graphs/   the figure above, and the script that generates it
results/         A/B metrics and the report
notes/           ship note for the current kernel set
docs/            methodology
```

Measurement artifacts are MIT. llama.cpp and the model weights keep their own licenses.
