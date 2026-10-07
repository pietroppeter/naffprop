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
import subprocess
import sys
from pathlib import Path

CASE = Path(__file__).with_name("benchmark_case.py")


def run(lib, n, damping, max_iter, conv):
    out = subprocess.run([sys.executable, CASE, lib, str(n), str(damping), str(max_iter), str(conv)],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


SIZES = [500, 1_000, 2_000, 4_000]
SETTINGS = [("sklearn defaults", 0.5, 200, 15), ("R defaults", 0.9, 1000, 100)]

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
