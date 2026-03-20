"""
CPU / memory monitoring utilities for process-level sampling.
"""

import subprocess
import threading


def read_process_cpu_rss(pid):
    proc = subprocess.run(
        ["ps", "-p", str(pid), "-o", "%cpu=", "-o", "rss="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    text = proc.stdout.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 2:
        return None
    try:
        cpu_pct = float(parts[0])
        rss_mb = int(parts[1]) / 1024.0
    except ValueError:
        return None
    return cpu_pct, rss_mb


class ProcessSampler:
    def __init__(self, pid, interval=0.5):
        self.pid = pid
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _run(self):
        while not self._stop.is_set():
            stats = read_process_cpu_rss(self.pid)
            if stats is not None:
                self.samples.append(stats)
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

