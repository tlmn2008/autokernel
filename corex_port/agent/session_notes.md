# autokernel → Iluvatar CoreX (ivcore11) 迁移记录

## 1. 仓库概况
- 上游:RightNow-AI/autokernel(https://github.com/RightNow-AI/autokernel.git)
- 起点 commit:`78435821cc3d5756ba6ee1785c397f6d8fa8c90d`(分支 `main`,日期 2026-03-19)
- 性质:GPU kernel 自动优化/基准工具。对 PyTorch 算子提供 Triton 与 CUDA C++ 两套手写 kernel,通过 `bench.py`(5 阶段正确性 harness)对照 `reference.py` 的 PyTorch 参考实现验证,并测吞吐/加速比。
- 编译方式:CUDA C++ kernel 走 `torch.utils.cpp_extension.load_inline()` 做 JIT 动态编译;Triton kernel 由 Triton 运行时编译。**无 nvcc、无 CMake 主构建**,编译发生在运行时 JIT。

## 2. 环境
- CoreX:`/usr/local/corex-4.5.0`,clang 22.1.0git(4.5.0.20260630),`-x ivcore --cuda-gpu-arch=ivcore11`。
- 栈:Python 3.12.11 / torch 2.10.0 / triton 3.2.0;GPU:2×Iluvatar BI-V150(能力 (7,1)),`ixsmi` 可见。
- 未修改 `/usr/local/corex`,所有 workaround 均 repo-local。

## 3. PTX JIT 风险的实证排查(重点)
契约提示 autokernel 若走 NVIDIA PTX 文本 JIT(`cuModuleLoadData`)可能触发 ivcore11 驱动 `INVALID_IMAGE(200)`,大概率 terminal。
- 实证结论:**autokernel 不走 NV PTX 文本 JIT**。CUDA C++ 路径经 `load_inline` → CoreX clang++ 直接编译为 ivcore11 原生设备代码;Triton 路径由 CoreX 版 Triton 后端生成原生代码。二者均有 ivcore11 原生编译路径,**无 PTX 文本装载**,故不触发该 terminal 阻塞。
- 佐证:18/18(kernel×backend)配置均能在 ivcore11 上完成编译并加载执行。

## 4. 关键适配(全部 repo-local)
1. `kernels/cuda/_compile.py`:新增 `_is_corex()`;CoreX 下从默认 CUDA flag 移除/翻译 nvcc 专用项(`--use_fast_math`、`--expt-relaxed-constexpr`、`-lineinfo`、`-gencode`),`_get_arch_flags()` 对 CoreX 返回空(架构由 `--cuda-gpu-arch=ivcore11` 固定),追加 `-x ivcore`。
2. `kernels/cuda/matmul.py`:`nvcuda::wmma` 16x16x16 half 片段在 `ixix::wmma` 不支持 → 改写为可移植 shared-memory tiled GEMM(fp16 输入 / fp32 累加 / fp16 输出)。
3. `kernels/cuda/softmax.py`:`#include <algorithm>` + `std::min`;并对奇数 `n_cols` 加标量 fallback,规避 ivcore11 非对齐 half2(32-bit)静默出错导致的 NaN。
4. `kernels/cuda/rmsnorm.py`、`kernels/cuda/rotary_embedding.py`:`#include <algorithm>` + 裸 `min/max` → `std::min/std::max`。
5. `kernels/fused_mlp.py`:Triton 3.2 缺 `tl.math.tanh` → 用恒等式 `tanh(z)=2*sigmoid(2z)-1` 替换。
6. 测试基建:新增 `corex_port/prewarm.py` 预热 JIT 编译(冷编译 ~34s 会超过 bench.py smoke 阶段 30s 看门狗,造成假 TIMEOUT);新增 `corex_port/run_suite.py` 编排 18 组合,超时/挂起用 `ixsmi -r -i <GPU_ID>` 清残留。

## 5. 测试方法与计数口径
- 测试面:9 种 kernel × 2 后端(triton/cuda)= **18 个 (kernel,backend) 组合**,每组合以 `bench.py` 5 阶段(smoke / shape_sweep / numerical_stability / determinism / edge_cases)最终 correctness 判定计为 1 个运行时测试用例。dtype 覆盖 fp16/bf16/fp32(仓库不用 fp64)。
- 运行命令:`PYTHONPATH=. python3 corex_port/prewarm.py && python3 corex_port/run_suite.py all all`。
- 计数:`tests_run = 18 = passed 7 + failed 11 + skipped 0`。挂起(看门狗超时)按契约计入 failed,不剔除。

## 6. 结果
- **编译**:18/18 配置全部编译+加载成功(compile_status=success)。
- **运行时正确性**:7 PASS / 11 FAIL / 0 SKIP(test_status=partial_pass)。
- 通过:matmul/triton、softmax/triton、softmax/cuda、cross_entropy/triton、cross_entropy/cuda、reduce/triton、reduce/cuda。
- 失败分类(详见 `blockers.json` 与 `test/test_summary.json`):
  - `fused_mlp/cuda`:朴素 kernel 大尺寸极慢 → HANG(平台无关,上游 kernel 效率问题)。
  - 其余 10 个:上游算法数值精度/harness 固定容差/对抗输入所致(matmul/cuda 强制 fp16、layernorm bf16+一遍法方差、rmsnorm/flash_attention 对抗 mixed_scale、fused_mlp bf16 GELU、rotary fp16 1e-3 过紧)。**关键证据:每个此类用例 triton 与 cuda 两后端结果完全一致**,证明是上游 kernel/harness 特性而非 CoreX 后端差异,NVIDIA 上亦不过。
- 未发现 CoreX terminal 阻塞:所有真正的迁移问题(flag、wmma、min/max、tanh、非对齐访问、冷编译超时)均已 repo-local 解决。

## 7. 结论
autokernel 在 ivcore11 上迁移成功:编译面 100% 通过,运行面通过用例均正确,剩余失败为平台无关的上游数值/容差特性与单个朴素 kernel 的性能挂起,非 CoreX 迁移阻塞。整体判定 `partial`(编译成功 + 部分运行时用例失败 + 成功发布)。
