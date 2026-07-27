#!/usr/bin/env python3
"""AutoKernel CoreX 迁移测试驱动。

对 9 种 kernel × 2 后端(triton / cuda)逐一运行 bench.py 的完整 5 阶段
正确性校验(smoke / shape_sweep / numerical_stability / determinism /
edge_cases),每个 (kernel, backend) 组合以 bench.py 输出的最终 `correctness:`
判定作为一个运行时测试用例。全部输出汇入 test.log。带超时看门狗,超时/崩溃
计为 failed,不剔除。
"""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

REPO = "/home/repos/autokernel"
OUT_DIR = os.path.join(REPO, "corex_port", "test")
TEST_LOG = os.path.join(OUT_DIR, "test.log")
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")

KERNELS = [
    "matmul", "softmax", "layernorm", "rmsnorm", "flash_attention",
    "fused_mlp", "cross_entropy", "rotary_embedding", "reduce",
]
BACKENDS = {
    "triton": "kernels/{k}.py",
    "cuda": "kernels/cuda/{k}.py",
}

PER_RUN_TIMEOUT = 420  # 秒:单个 bench.py 运行的看门狗

STAGE_KEYS = ["smoke_test", "shape_sweep", "numerical_stability",
              "determinism", "edge_cases"]


def parse_output(text):
    """从 bench.py 输出解析各阶段与最终 correctness / 性能指标。"""
    res = {k: None for k in STAGE_KEYS}
    res["correctness"] = None
    res["throughput_tflops"] = None
    res["speedup_vs_pytorch"] = None
    # 只取 '--- Correctness Summary ---' 之后的阶段行
    summ = text
    idx = text.rfind("--- Correctness Summary ---")
    if idx != -1:
        summ = text[idx:]
    for k in STAGE_KEYS:
        m = re.search(rf"^{k}:\s*(\S+)", summ, re.MULTILINE)
        if m:
            res[k] = m.group(1)
    # 最终 correctness 取 '=== FINAL ===' 段的值
    fi = text.rfind("=== FINAL ===")
    tail = text[fi:] if fi != -1 else text
    m = re.search(r"^correctness:\s*(\S+)", tail, re.MULTILINE)
    if m:
        res["correctness"] = m.group(1)
    m = re.search(r"^throughput_tflops:\s*([\d.]+)", tail, re.MULTILINE)
    if m:
        res["throughput_tflops"] = float(m.group(1))
    m = re.search(r"^speedup_vs_pytorch:\s*([\d.]+)x", tail, re.MULTILINE)
    if m:
        res["speedup_vs_pytorch"] = float(m.group(1))
    return res


def gpu_reset():
    for dev in ("0", "1"):
        try:
            subprocess.run(["ixsmi", "-r", "-i", dev], timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def run_one(kernel, backend, logf):
    src = os.path.join(REPO, BACKENDS[backend].format(k=kernel))
    dst = os.path.join(REPO, "kernel.py")
    header = f"\n{'#'*78}\n# CONFIG: kernel={kernel} backend={backend}\n# source={src}\n{'#'*78}\n"
    logf.write(header)
    logf.flush()
    if not os.path.exists(src):
        logf.write(f"SOURCE MISSING: {src}\n")
        return {"kernel": kernel, "backend": backend, "correctness": "MISSING",
                "stages": {}, "throughput_tflops": None, "speedup_vs_pytorch": None,
                "exit": None, "elapsed_s": 0.0}
    shutil.copyfile(src, dst)
    cmd = [sys.executable, "bench.py"]
    t0 = time.time()
    timed_out = False
    try:
        p = subprocess.run(cmd, cwd=REPO, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, timeout=PER_RUN_TIMEOUT,
                           text=True)
        out = p.stdout
        rc = p.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") if isinstance(e.stdout, str) else (
            e.stdout.decode("utf-8", "replace") if e.stdout else "")
        out += f"\n\n*** WATCHDOG TIMEOUT after {PER_RUN_TIMEOUT}s -> HUNG ***\n"
        rc = -9
        timed_out = True
    elapsed = time.time() - t0
    logf.write(out)
    logf.write(f"\n[exit={rc} elapsed={elapsed:.1f}s]\n")
    logf.flush()

    parsed = parse_output(out)
    correctness = parsed["correctness"]
    if timed_out:
        correctness = "HANG"
    elif correctness is None:
        correctness = "FAIL"  # 崩溃/无输出
    if timed_out:
        gpu_reset()
    return {
        "kernel": kernel, "backend": backend,
        "correctness": correctness,
        "stages": {k: parsed[k] for k in STAGE_KEYS},
        "throughput_tflops": parsed["throughput_tflops"],
        "speedup_vs_pytorch": parsed["speedup_vs_pytorch"],
        "exit": rc, "elapsed_s": round(elapsed, 1),
    }


def main():
    only_k = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else KERNELS
    only_b = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] != "all" else list(BACKENDS)
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []
    mode = "a" if (len(sys.argv) > 3 and sys.argv[3] == "append") else "w"
    with open(TEST_LOG, mode) as logf:
        logf.write(f"\n===== AutoKernel CoreX 测试运行 @ {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for k in only_k:
            for b in only_b:
                print(f">>> running {k} / {b} ...", flush=True)
                r = run_one(k, b, logf)
                print(f"    -> correctness={r['correctness']} "
                      f"tflops={r['throughput_tflops']} exit={r['exit']} "
                      f"{r['elapsed_s']}s", flush=True)
                results.append(r)
    # 合并/更新 results.json
    prev = []
    if mode == "a" and os.path.exists(RESULTS_JSON):
        try:
            prev = json.load(open(RESULTS_JSON))
        except Exception:
            prev = []
    # 用新结果覆盖同 (kernel,backend)
    key = lambda r: (r["kernel"], r["backend"])
    merged = {key(r): r for r in prev}
    for r in results:
        merged[key(r)] = r
    final = [merged[key(r)] for r in sorted(merged.values(), key=lambda r: (KERNELS.index(r["kernel"]) if r["kernel"] in KERNELS else 99, r["backend"]))]
    json.dump(final, open(RESULTS_JSON, "w"), indent=2, ensure_ascii=False)
    # 汇总
    npass = sum(1 for r in results if r["correctness"] == "PASS")
    print(f"\nDONE this run: {npass}/{len(results)} PASS")
    for r in results:
        print(f"  {r['kernel']:<18} {r['backend']:<7} {r['correctness']}")


if __name__ == "__main__":
    main()
