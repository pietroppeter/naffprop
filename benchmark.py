# /// script
# requires-python = ">=3.9"
# dependencies = ["naffprop", "numpy", "scikit-learn"]
#
# [tool.uv.sources]
# naffprop = { path = "." }
# ///
"""Time naffprop against scikit-learn's AffinityPropagation, on the same
similarity matrix with the same parameters, and measure the memory each
needs on top of that matrix.

    uv run benchmark.py

Every case runs benchmark_case.py in its own process, so the peak memory
(max RSS) of one does not hide the next one. Unix only (ru_maxrss).
"""

import json
import os
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

CASE = Path(__file__).with_name("benchmark_case.py")


def run(lib, n, damping, max_iter, conv):
    out = subprocess.run([sys.executable, CASE, lib, str(n), str(damping), str(max_iter), str(conv)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a["labels"], b["labels"])


def sh(*cmd):
    try:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def machine():
    """The machine and software the timings depend on, to paste with them."""
    cpu, ram = platform.processor() or platform.machine(), None
    if sys.platform == "darwin":
        cpu = sh("sysctl", "-n", "machdep.cpu.brand_string") or cpu
        ram = int(sh("sysctl", "-n", "hw.memsize") or 0)
    elif sys.platform.startswith("linux"):
        try:
            info = Path("/proc/cpuinfo").read_text()
            cpu = next((l.split(":", 1)[1].strip() for l in info.splitlines()
                        if l.startswith("model name")), cpu)
            mem = Path("/proc/meminfo").read_text().split()
            ram = int(mem[mem.index("MemTotal:") + 1]) * 1024
        except (OSError, ValueError):
            pass
    mac = platform.mac_ver()[0]
    os_name = f"macOS {mac}" if mac else platform.platform(terse=True)
    lines = [
        f"- CPU: {cpu}, {os.cpu_count()} cores" + (f", {ram / 2**30:.0f} GiB RAM" if ram else ""),
        f"- OS: {os_name} ({platform.machine()})",
        f"- Python {platform.python_version()}, naffprop {version('naffprop')}, "
        f"numpy {version('numpy')}, scikit-learn {version('scikit-learn')}",
    ]
    return "\n".join(lines)


SIZES = [500, 1_000, 2_000, 4_000]
SETTINGS = [("sklearn defaults", 0.5, 200, 15), ("R defaults", 0.9, 1000, 100)]

print(machine() + "\n")
print("Time (s) and extra peak memory (MiB, on top of the n x n similarity matrix).\n")
print("| n | parameters | naffprop | scikit-learn | speedup | naffprop memory | scikit-learn memory | clusters | iterations |")
print("|--:|:-----------|---------:|-------------:|--------:|----------------:|--------------------:|---------:|-----------:|")
for n in SIZES:
    for name, damping, max_iter, conv in SETTINGS:
        a = run("naffprop", n, damping, max_iter, conv)
        b = run("sklearn", n, damping, max_iter, conv)
        print(f"| {n:,} | {name} | {a['time']:.2f} | {b['time']:.2f} | {b['time'] / a['time']:.1f}x "
              f"| {a['mem']:.0f} | {b['mem']:.0f} | {a['k']} / {b['k']} | {a['its']} / {b['its']} |",
              flush=True)

print("\nnaffprop's memory options, with R's defaults: float64 (the default), float32 "
      "(dtype=np.float32), and float32 without a copy of the similarities (copy=False). "
      "Time (s) and extra peak memory (MiB).\n")
print("| n | float64 | float32 | float32, copy=False | float64 memory | float32 memory | float32, copy=False memory |")
print("|--:|--------:|--------:|--------------------:|---------------:|---------------:|---------------------------:|")
for n in SIZES:
    r = [run(lib, n, 0.9, 1000, 100) for lib in ("naffprop", "naffprop-f32", "naffprop-f32-inplace")]
    print(f"| {n:,} | " + " | ".join(f"{x['time']:.2f}" for x in r) + " | "
          + " | ".join(f"{x['mem']:.0f}" for x in r) + " |", flush=True)

LARGE = [4_000, 8_000]
print("\nFor large n, with R's defaults: sparse similarities (the 10% nearest neighbours of each "
      "point) and leveraged AP (10% of the points, 5 sweeps), against the full float64 run. "
      "Time (s), memory (MiB) including the similarities, and agreement with the full "
      "clustering (adjusted Rand index).\n")
print("| n | full | sparse | leveraged | full memory | sparse memory | leveraged memory "
      "| clusters | ARI sparse | ARI leveraged |")
print("|--:|-----:|-------:|----------:|------------:|--------------:|-----------------:"
      "|---------:|-----------:|--------------:|")
for n in LARGE:
    full, sp, lev = (run(lib, n, 0.9, 1000, 100)
                     for lib in ("naffprop", "naffprop-sparse", "naffprop-leveraged"))
    full_mem = full["mem"] + n * n * 8 / 2**20
    sp_mem = sp["mem"] + (n * (n // 10)) * 12 / 2**20  # float64 values, int32 columns
    print(f"| {n:,} | {full['time']:.2f} | {sp['time']:.2f} | {lev['time']:.2f} | {full_mem:.0f} "
          f"| {sp_mem:.0f} | {lev['mem']:.0f} | {full['k']} / {sp['k']} / {lev['k']} "
          f"| {ari(full, sp):.2f} | {ari(full, lev):.2f} |", flush=True)

print("\nScaleAP's pruning (scaleap=True), with R's defaults, against the dense loop, in float64 and "
      "in float32 with copy=False. Time (s), extra peak memory (MiB), and whether the clusters "
      "are the same.\n")
print("| n | dense | scaleap | speedup | dense memory | scaleap memory | float32 dense | float32 scaleap "
      "| float32 dense memory | float32 scaleap memory | same clusters |")
print("|--:|------:|--------:|--------:|-------------:|---------------:|--------------:|----------------:"
      "|---------------------:|-----------------------:|:-------------:|")
for n in [2_000, 4_000, 8_000]:
    d, s, d32, s32 = (run(lib, n, 0.9, 1000, 100) for lib in (
        "naffprop", "naffprop-scaleap", "naffprop-f32-inplace", "naffprop-f32-inplace-scaleap"))
    same = d["labels"] == s["labels"] and d32["labels"] == s32["labels"]
    print(f"| {n:,} | {d['time']:.2f} | {s['time']:.2f} | {d['time'] / s['time']:.1f}x "
          f"| {d['mem']:.0f} | {s['mem']:.0f} | {d32['time']:.2f} | {s32['time']:.2f} "
          f"| {d32['mem']:.0f} | {s32['mem']:.0f} | {'yes' if same else 'no'} |", flush=True)
