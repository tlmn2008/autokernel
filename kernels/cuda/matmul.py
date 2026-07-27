"""
AutoKernel -- CUDA C++ Matrix Multiplication kernel.

CoreX (ivcore11) port note:
  The upstream kernel used NVIDIA tensor cores via nvcuda::wmma with a
  <16,16,16,__half> fragment shape.  On Iluvatar ivcore11 the wmma header
  (ixix::wmma) does NOT provide the __half 16x16x16 fragment specialisation
  (it ships __half tiles of 16x16x32 / 32x32x16 with a float accumulator, and
  a separate float 16x16x16 shape).  To keep this kernel correct on CoreX
  without depending on an NVIDIA-only tensor-core tile shape, the tensor-core
  path is replaced with a portable shared-memory tiled GEMM (fp16 inputs,
  fp32 accumulation).  Same launcher signature / semantics.

Target metric: throughput_tflops (higher is better)
Secondary: correctness must ALWAYS pass
"""

KERNEL_TYPE = "matmul"
BACKEND = "cuda"

import torch
from kernels.cuda._compile import compile_cuda

CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Portable shared-memory tiled GEMM: C[MxN] = A[MxK] @ B[KxN], fp16 in/out,
// fp32 accumulation.  No tensor-core / wmma dependency, so it runs on
// ivcore11 as well as NVIDIA.
constexpr int TILE = 16;

__global__ void matmul_kernel_tiled(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;

    const int k_tiles = (K + TILE - 1) / TILE;
    for (int kt = 0; kt < k_tiles; ++kt) {
        const int a_col = kt * TILE + threadIdx.x;
        const int b_row = kt * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : __float2half(0.0f);

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2half(acc);
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "A must be float16");
    TORCH_CHECK(B.dtype() == torch::kFloat16, "B must be float16");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    matmul_kernel_tiled<<<grid, block>>>(
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        M, N, K
    );

    return C;
}
"""

_module = None


def _get_module():
    global _module
    if _module is None:
        _module = compile_cuda(CUDA_SRC, "matmul_cuda")
    return _module


def kernel_fn(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Entry point called by bench.py. Must match reference.matmul_ref signature."""
    assert A.is_cuda and B.is_cuda

    # Handle non-fp16 inputs by casting
    orig_dtype = A.dtype
    if A.dtype != torch.float16:
        A = A.to(torch.float16)
    if B.dtype != torch.float16:
        B = B.to(torch.float16)

    mod = _get_module()
    C = mod.matmul_cuda(A, B)

    # Cast back if needed
    if orig_dtype != torch.float16:
        C = C.to(orig_dtype)

    return C
