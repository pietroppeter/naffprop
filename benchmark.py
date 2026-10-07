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
