"""Parallel, fault-isolated execution of untrusted CadQuery scripts.

Every script runs in its own short-lived process forked from a warm zygote (zygote.py):

* hard wall-clock timeout (the whole process group is SIGKILLed),
* a segfault inside the CAD kernel only loses that one sample,
* memory / CPU-time / file-size rlimits, no core dumps,
* empty network namespace (when unprivileged user namespaces are available),
* private scratch working directory, removed afterwards,
* a static check that refuses scripts importing subprocess/socket/shutil/... before they run.

This is defence in depth against *accidents* in model-written code, not a jail: scripts
can still read files and write wherever the user can. Do not point it at adversarial input.
"""

from __future__ import annotations

import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
from collections.abc import Sequence

from . import codeutil


def default_workers() -> int:
    env = os.environ.get("T2C_CAD_WORKERS")
    if env:
        return max(1, int(env))
    try:
        n = len(os.sched_getaffinity(0))  # honours Slurm/cgroup cpusets
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, n - 1)


class CadExecutor:
    """``run(jobs)`` -> list of result dicts in the same order (schema: ``worker.run_job``)."""

    def __init__(self, workers: int | None = None, timeout: float = 60.0, mem_gb: float | None = None):
        self.workers = workers or default_workers()
        self.timeout = float(timeout)
        gb = float(mem_gb if mem_gb is not None else os.environ.get("T2C_SANDBOX_MEM_GB", 16))
        self.mem_bytes = int(gb * (1 << 30))
        self.info: dict = {}
        self._proc: subprocess.Popen | None = None
        self._req = self._res = None
        self._results: queue.Queue = queue.Queue()
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> CadExecutor:
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return self
            req_r, req_w = os.pipe()
            res_r, res_w = os.pipe()
            env = dict(os.environ)
            src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env["PYTHONPATH"] = os.pathsep.join(p for p in (src_root, env.get("PYTHONPATH")) if p)
            self._proc = subprocess.Popen(
                [sys.executable, "-m", "t2c.cad.zygote", str(req_r), str(res_w)],
                pass_fds=(req_r, res_w),
                stdin=subprocess.DEVNULL,
                stdout=sys.stderr,  # stray prints from native libs must not reach our pipes
                env=env,
                start_new_session=True,  # a Ctrl-C in the terminal reaches us, not the zygote
            )
            os.close(req_r)
            os.close(res_w)
            self._req, self._res = os.fdopen(req_w, "wb", buffering=0), os.fdopen(res_r, "rb", buffering=0)
            # Results are drained by a thread so the zygote never blocks on a full pipe while
            # we are blocked writing requests to it (classic two-pipe deadlock).
            self._results = queue.Queue()
            threading.Thread(target=self._reader, args=(self._res, self._results), daemon=True).start()
            probe = self._dispatch([{"probe": True}], timeout=300.0)[0]
            self.info = probe
            if probe.get("cq_import_error") or probe.get("error"):
                self._stop()
                raise RuntimeError(f"CAD worker failed to start: {probe}")
        return self

    def _stop(self) -> None:
        for fh in (self._req, self._res):
            try:
                if fh is not None:
                    fh.close()
            except OSError:
                pass
        if self._proc is not None:
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = self._req = self._res = None

    def close(self) -> None:
        with self._lock:
            self._stop()

    def __enter__(self) -> CadExecutor:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self._stop()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ API
    def run(self, jobs: Sequence[dict], progress: str | None = None) -> list[dict]:
        """Score a batch. Jobs with no code, or rejected statically, never reach a process."""
        self.start()
        results: list[dict | None] = [None] * len(jobs)
        todo: list[int] = []
        for i, job in enumerate(jobs):
            code = job.get("code")
            if not code or not code.strip():
                results[i] = {"ok": False, "stage": 0, "error_type": "no_code", "error": "no Python code block found in the response"}
                continue
            bad = codeutil.static_check(code)
            if bad:
                results[i] = {"ok": False, "stage": 0, "error_type": bad[0], "error": bad[1]}
                continue
            todo.append(i)
        if todo:
            with self._lock:
                for i, res in zip(todo, self._dispatch([jobs[i] for i in todo], progress=progress), strict=True):
                    results[i] = res
        return results  # type: ignore[return-value]

    def run_one(self, job: dict) -> dict:
        return self.run([job])[0]

    # ------------------------------------------------------------------ internals
    def _send(self, job_id: int, job: dict, timeout: float) -> None:
        payload = pickle.dumps((job_id, job, timeout, self.mem_bytes), protocol=pickle.HIGHEST_PROTOCOL)
        self._req.write(struct.pack(">I", len(payload)) + payload)

    @staticmethod
    def _reader(fh, out: queue.Queue) -> None:
        def read_exact(n: int) -> bytes:
            buf = bytearray()
            while len(buf) < n:
                chunk = fh.read(n - len(buf))
                if not chunk:
                    raise EOFError
                buf += chunk
            return bytes(buf)

        try:
            while True:
                (n,) = struct.unpack(">I", read_exact(4))
                out.put(pickle.loads(read_exact(n)))
        except Exception as e:  # noqa: BLE001 - EOF on shutdown, or the zygote died
            out.put(e)

    def _recv(self) -> tuple[int, dict]:
        item = self._results.get()
        if isinstance(item, Exception):
            raise RuntimeError("the CAD zygote process died (see stderr above)") from item
        return item

    def _dispatch(self, jobs: list[dict], timeout: float | None = None, progress: str | None = None) -> list[dict]:
        results: list[dict | None] = [None] * len(jobs)
        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=len(jobs), desc=progress, smoothing=0.05, mininterval=2.0)
        nxt = active = 0
        try:
            while nxt < len(jobs) or active:
                while nxt < len(jobs) and active < self.workers:
                    self._send(nxt, jobs[nxt], timeout or float(jobs[nxt].get("timeout") or self.timeout))
                    nxt += 1
                    active += 1
                job_id, res = self._recv()
                results[job_id] = res
                active -= 1
                if bar is not None:
                    bar.update(1)
        except BaseException:
            self._stop()  # closing the request pipe makes the zygote kill its children and exit
            raise
        finally:
            if bar is not None:
                bar.close()
        return results  # type: ignore[return-value]
