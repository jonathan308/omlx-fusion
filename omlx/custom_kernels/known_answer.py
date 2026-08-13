# SPDX-License-Identifier: Apache-2.0
"""Kernel known-answer gate for the MLX build oMLX is running on.

Catches the corrupt-Metal-kernel failure mode of single-stage wheel builds
(token salad at normal t/s): every check computes the same op on the GPU
stream and the CPU stream and requires agreement. The 4-bit quantized matmul
path is the one the quantized models oMLX serves actually exercise — it is
the check that matters most, including for the custom kernels bundled in
this package.

Run on every node after any wheel swap, before soaks:

    python -m omlx.custom_kernels.known_answer

Exit 0 = pass. The admin console's cluster tab can also run it in-process
(``POST /admin/api/cluster/known-answer``).
"""

from __future__ import annotations

import sys
from typing import Any


def _close(a: Any, b: Any, rtol: float, name: str, report: list[str], mx: Any) -> bool:
    """Relative agreement: corrupt kernels produce O(1) relative garbage;
    legit GPU-vs-CPU accumulation-order differences sit orders below these
    thresholds. Absolute tolerances proved too tight across GPU generations
    (one M-series rank accumulates differently than another at fp32)."""

    a32, b32 = a.astype(mx.float32), b.astype(mx.float32)
    max_diff = mx.max(mx.abs(a32 - b32)).item()
    ref = max(mx.max(mx.abs(b32)).item(), 1e-6)
    rel = max_diff / ref
    report.append(
        f"    {name}: max_abs_diff={max_diff:.3e} rel={rel:.3e} (rtol {rtol})"
    )
    return rel < rtol


def run_checks(mx: Any | None = None) -> dict[str, Any]:
    """Run every check and return a JSON-serializable verdict.

    ``mx`` is injectable so tests can drive the gate with a stub; production
    callers leave it None and the real ``mlx.core`` is imported lazily, so
    importing this module never pulls MLX into a process that does not need
    it.
    """

    if mx is None:
        import mlx.core as mx

    mx.random.seed(7)
    failures: list[str] = []
    report: list[str] = []

    # 1. plain matmul fp32: gpu vs cpu (rtol generous: catches garbage, not
    #    rounding)
    a = mx.random.normal((256, 512))
    b = mx.random.normal((512, 128))
    g = mx.matmul(a, b, stream=mx.gpu)
    c = mx.matmul(a, b, stream=mx.cpu)
    mx.eval(g, c)
    if not _close(g, c, 1e-3, "matmul_fp32", report, mx):
        failures.append("matmul_fp32 gpu/cpu divergence")

    # 2. 4-bit quantized matmul (the quantized-model hot path)
    # transpose=True (default): w is (out, in), computes x @ w.T
    w = mx.random.normal((1024, 512))
    wq, scales, biases = mx.quantize(w, bits=4)
    x = mx.random.normal((8, 512))
    g = mx.quantized_matmul(x, wq, scales, biases, bits=4, stream=mx.gpu)
    c = mx.quantized_matmul(x, wq, scales, biases, bits=4, stream=mx.cpu)
    mx.eval(g, c)
    if not _close(g, c, 1e-2, "qmm_4bit", report, mx):
        failures.append("quantized_matmul_4bit gpu/cpu divergence")

    # 3. softmax -> argmax chain (decode head path) — exact index agreement
    logits = mx.random.normal((4, 32000))
    g = mx.argmax(mx.softmax(logits, axis=-1), axis=-1)
    c = mx.argmax(
        mx.softmax(logits.astype(mx.float32), axis=-1, stream=mx.cpu),
        axis=-1,
        stream=mx.cpu,
    )
    mx.eval(g, c)
    agree = int(mx.sum(g == c).item())
    report.append(f"    softmax/argmax: {agree}/4 rows agree")
    if agree < 4:
        failures.append("softmax/argmax gpu/cpu divergence")

    # 4. fp16 layernorm-ish chain (norm + scale + residual)
    h = mx.random.normal((64, 2048)).astype(mx.float16)

    def norm_chain(t: Any, stream: Any) -> Any:
        m = mx.mean(t, axis=-1, keepdims=True, stream=stream)
        v = mx.var(t, axis=-1, keepdims=True, stream=stream)
        return ((t - m) * mx.rsqrt(v + 1e-5, stream=stream) + t).astype(mx.float32)

    g = norm_chain(h, mx.gpu)
    c = norm_chain(h.astype(mx.float32), mx.cpu)
    mx.eval(g, c)
    if not _close(g, c, 5e-2, "norm_chain_fp16", report, mx):
        failures.append("norm_chain fp16 gpu/cpu divergence")

    return {
        "ok": not failures,
        "mlx_version": str(getattr(mx, "__version__", "unknown")),
        "report": report,
        "failures": failures,
    }


def main() -> int:
    result = run_checks()
    print(
        f"mlx {result['mlx_version']} known-answer: "
        f"{'PASS' if result['ok'] else 'FAIL'}"
    )
    for line in result["report"]:
        print(line)
    for failure in result["failures"]:
        print(f"  FAILED: {failure}")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
