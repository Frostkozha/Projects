"""Resource probes: actual VRAM, RAM and disk (Brain plan v0.3, section 3). Metadata only."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def gpu_memory() -> Optional[dict]:
    """Total/used/free VRAM in MiB plus driver and name, from nvidia-smi; None when unavailable."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10,
                             shell=False).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    name, total, used, free, driver = [x.strip() for x in out[0].split(",")]
    return {"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(free), "driver": driver}


def system_memory() -> Optional[dict]:
    if sys.platform == "win32":
        import ctypes  # noqa: PLC0415

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = MEMORYSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        mib = 1024 * 1024
        return {"total_mib": st.ullTotalPhys // mib, "available_mib": st.ullAvailPhys // mib,
                "commit_limit_mib": st.ullTotalPageFile // mib,
                "commit_available_mib": st.ullAvailPageFile // mib}
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        kib = lambda k: int(info[k].strip().split()[0]) // 1024  # noqa: E731
        return {"total_mib": kib("MemTotal"), "available_mib": kib("MemAvailable"),
                "swap_total_mib": kib("SwapTotal"), "swap_free_mib": kib("SwapFree")}
    except (OSError, KeyError, ValueError):
        return None


def disk_free_mib(path: Path) -> int:
    return shutil.disk_usage(path).free // (1024 * 1024)


def snapshot(path: Path) -> dict:
    return {"gpu": gpu_memory(), "ram": system_memory(), "disk_free_mib": disk_free_mib(path),
            "platform": sys.platform}
