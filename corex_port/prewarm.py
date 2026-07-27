#!/usr/bin/env python3
"""预热所有 CUDA kernel 的 JIT 编译到磁盘缓存。

torch.utils.cpp_extension.load_inline 的冷编译约 ~34s,会超过 bench.py smoke
阶段内置的 _Timeout(30),导致「编译中被判 TIMEOUT」的假失败。先在此把每个
CUDA 模块编译并缓存(~/.cache/autokernel/cuda_build),bench.py 后续运行即可
直接加载 .so,反映真实运行时正确性。不修改测试 harness 本身。
"""
import importlib, time, traceback, torch, bench

KERNELS = ["matmul", "softmax", "layernorm", "rmsnorm", "flash_attention",
           "fused_mlp", "cross_entropy", "rotary_embedding", "reduce"]

for k in KERNELS:
    mod = importlib.import_module(f"kernels.cuda.{k}")
    cfg = bench.KERNEL_CONFIGS[k]
    label, sz = cfg["test_sizes"][0]
    dt = cfg["test_dtypes"][0]
    inp = cfg["input_generator"](sz, dt, "cuda", seed=42)
    t0 = time.time()
    try:
        out = mod.kernel_fn(**inp)
        torch.cuda.synchronize()
        print(f"{k:<18} OK   {time.time()-t0:6.1f}s")
    except Exception as e:
        el = [l for l in str(e).splitlines() if 'error:' in l]
        print(f"{k:<18} FAIL {time.time()-t0:6.1f}s  {(el[0] if el else str(e).splitlines()[0])[:120]}")
