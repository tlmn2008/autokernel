#!/usr/bin/env python3
"""非侵入式守卫验证 (autokernel corex-port-guarded 分支)。

本环境无 NVIDIA 卡,无法在真机跑 NVIDIA 回归,故用两种走查证明
"默认(非 CoreX)路径保持上游原样":

  A. 构建/编译选项 (_compile.py):对 `_default_cuda_flags(is_corex)` 与
     `_get_arch_flags(is_corex)` 做单元断言 —— 非 CoreX 分支必须下发原始
     nvcc flag(--use_fast_math / -lineinfo / --expt-relaxed-constexpr /
     -gencode),CoreX 分支才翻译为 clang flag 并置 -DAUTOKERNEL_COREX。

  B. 源码守卫 (matmul.py / softmax.py):用 CoreX clang++ 预处理器对 CUDA_SRC
     分别在 "未定义 AUTOKERNEL_COREX"(= NVIDIA 默认)与 "定义 AUTOKERNEL_COREX"
     (= CoreX) 两种情况下展开 #if/#else/#endif,断言:
       - 未定义时选中原始 nvcuda::wmma / vectorized-half2 路径,且不含 CoreX 变体;
       - 定义时选中 CoreX 变体(tiled GEMM / 奇数列标量 softmax),且不含 NVIDIA 变体。

用法: python3 corex_port/test/test_guard_noninvasive.py
"""
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

CLANGXX = os.environ.get("CXX") or "/usr/local/corex/bin/clang++"

_fail = []


def check(name, cond):
    print(f"[{'OK ' if cond else 'FAIL'}] {name}")
    if not cond:
        _fail.append(name)


def preprocess(src: str, define_corex: bool) -> str:
    """跑 clang++ 预处理器展开守卫。剥离 #include(守卫逻辑不依赖头文件)。"""
    stripped = "\n".join(
        l for l in src.splitlines() if not l.lstrip().startswith("#include")
    )
    with tempfile.NamedTemporaryFile("w", suffix=".cpp", delete=False) as f:
        f.write(stripped)
        path = f.name
    try:
        cmd = [CLANGXX, "-E", "-P", "-x", "c++"]
        if define_corex:
            cmd.append("-DAUTOKERNEL_COREX=1")
        cmd.append(path)
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stderr, file=sys.stderr)
            raise RuntimeError("clang++ -E failed")
        return out.stdout
    finally:
        os.unlink(path)


def main():
    # ---- A. 构建选项守卫 (_compile.py) ----
    from kernels.cuda import _compile

    nv = _compile._default_cuda_flags(False)
    cx = _compile._default_cuda_flags(True)
    print("NVIDIA default flags :", nv)
    print("CoreX  default flags :", cx)

    # 非 CoreX = 原始 nvcc flag,不含任何 CoreX 痕迹
    check("A1 NVIDIA flags 保留 --use_fast_math", "--use_fast_math" in nv)
    check("A2 NVIDIA flags 保留 -lineinfo", "-lineinfo" in nv)
    check("A3 NVIDIA flags 保留 --expt-relaxed-constexpr", "--expt-relaxed-constexpr" in nv)
    check("A4 NVIDIA flags 不含 -x ivcore", "ivcore" not in nv)
    check("A5 NVIDIA flags 不含 -DAUTOKERNEL_COREX",
          not any("AUTOKERNEL_COREX" in x for x in nv))
    # CoreX = 翻译后的 clang flag
    check("A6 CoreX flags 含 -DAUTOKERNEL_COREX=1", "-DAUTOKERNEL_COREX=1" in cx)
    check("A7 CoreX flags 含 -x ivcore", "ivcore" in cx)
    check("A8 CoreX flags 去除 nvcc 专用 flag",
          "--use_fast_math" not in cx and "-lineinfo" not in cx
          and "--expt-relaxed-constexpr" not in cx)

    # arch flag:非 CoreX 走 -gencode;CoreX 返回空
    nv_arch = _compile._get_arch_flags(is_corex=False)
    cx_arch = _compile._get_arch_flags(is_corex=True)
    print("NVIDIA arch flags    :", nv_arch)
    print("CoreX  arch flags    :", cx_arch)
    check("A9 NVIDIA arch 含 -gencode", any("-gencode" in x for x in nv_arch))
    check("A10 CoreX arch 为空", cx_arch == [])

    # ---- B. 源码守卫预处理走查 ----
    from kernels.cuda import matmul, softmax

    mm_nv = preprocess(matmul.CUDA_SRC, define_corex=False)
    mm_cx = preprocess(matmul.CUDA_SRC, define_corex=True)
    check("B1 matmul 默认(NVIDIA)选中 nvcuda::wmma 内核",
          "matmul_kernel_wmma" in mm_nv and "nvcuda" in mm_nv)
    check("B2 matmul 默认不含 CoreX tiled 变体",
          "matmul_kernel_tiled" not in mm_nv)
    check("B3 matmul CoreX 选中 tiled 变体",
          "matmul_kernel_tiled" in mm_cx)
    check("B4 matmul CoreX 不含 wmma 内核",
          "matmul_kernel_wmma" not in mm_cx and "nvcuda" not in mm_cx)

    sm_nv = preprocess(softmax.CUDA_SRC, define_corex=False)
    sm_cx = preprocess(softmax.CUDA_SRC, define_corex=True)
    # 预处理后注释被剥离,故用代码级标记判定:
    #   - 原始 vectorized 路径含 "n_tail && lane == 0" 的尾元素处理;
    #   - CoreX 标量 fallback 含按列步进 "c += 32"(标量循环,仅存在于该分支)。
    scalar_marker = "c += 32"
    check("B5 softmax 默认(NVIDIA)保留 half2 尾元素路径",
          "n_tail && lane == 0" in sm_nv)
    check("B6 softmax 默认不含 CoreX 标量 fallback",
          scalar_marker not in sm_nv)
    check("B7 softmax CoreX 含奇数列标量 fallback",
          scalar_marker in sm_cx)
    check("B8 softmax CoreX 不含 half2 尾元素路径(改为 n_tail==0 分支)",
          "n_tail && lane == 0" not in sm_cx)

    print()
    if _fail:
        print(f"FAILED: {len(_fail)} 项: {_fail}")
        sys.exit(1)
    print("ALL GUARD CHECKS PASSED — 默认路径保持上游 nvcc/wmma 原样,CoreX 变体仅在守卫内生效。")


if __name__ == "__main__":
    main()
