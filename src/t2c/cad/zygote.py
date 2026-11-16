"""Zygote: a small server process that imports the CAD stack once and forks one
throw-away child per script.

Why not ``multiprocessing``: its fork server re-imports the caller's ``__main__`` in every
child (i.e. torch + the whole training script per CAD job) and is a process-global
singleton other libraries may have started already. This process is ours alone.

Started by ``sandbox.CadExecutor`` as ``python -m t2c.cad.zygote <req_fd> <res_fd>``.
Wire format, both directions: 4-byte big-endian length + pickle.

    request   (job_id, job, timeout_s, mem_bytes)
    response  (job_id, result_dict)

The zygote owns timeouts, process-group kills, scratch directories and child reaping, so
a parent that dies (EOF on the request pipe) leaves nothing behind.
"""

from __future__ import annotations

import os
import sys

# --- 1. network isolation: must happen while this process is still single-threaded ------
NET_ISOLATED = False
NET_ERROR: str | None = None


def _isolate_network() -> None:
    """Enter an empty network namespace through an unprivileged user namespace.

    uid/gid are mapped to themselves, so file ownership and permissions are unchanged;
    children forked later inherit the namespace and cannot open network connections.
    """
    global NET_ISOLATED, NET_ERROR
    import ctypes

    CLONE_NEWUSER, CLONE_NEWNET = 0x10000000, 0x40000000
    uid, gid = os.getuid(), os.getgid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.unshare(CLONE_NEWUSER | CLONE_NEWNET) != 0:
            raise OSError(ctypes.get_errno(), os.strerror(ctypes.get_errno()))
        with open("/proc/self/setgroups", "w") as fh:
            fh.write("deny")
        with open("/proc/self/uid_map", "w") as fh:
            fh.write(f"{uid} {uid} 1")
        with open("/proc/self/gid_map", "w") as fh:
            fh.write(f"{gid} {gid} 1")
        NET_ISOLATED = True
    except Exception as e:  # noqa: BLE001 - user namespaces can be disabled by the admin
        NET_ERROR = f"{type(e).__name__}: {e}"


def main(argv: list[str]) -> int:
    req_fd, res_fd = int(argv[1]), int(argv[2])
    if os.environ.get("T2C_SANDBOX_NET", "1") != "0":
        _isolate_network()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"  # the pool already runs one process per core

    # --- 2. warm imports (seconds, paid once) ---------------------------------------------
    import pickle
    import selectors
    import shutil
    import signal
    import struct
    import tempfile
    import time

    import scipy.spatial  # noqa: F401
    import trimesh  # noqa: F401

    from t2c.cad import worker

    tmp_root = os.environ.get("T2C_TMPDIR") or tempfile.gettempdir()
    info = {"warm": True, "net_isolated": NET_ISOLATED, "net_error": NET_ERROR, "zygote_pid": os.getpid()}

    os.set_blocking(req_fd, False)
    sel = selectors.DefaultSelector()
    sel.register(req_fd, selectors.EVENT_READ, None)
    inbuf = bytearray()
    children: dict[int, dict] = {}  # read-fd -> state

    def send(job_id, result: dict) -> None:
        payload = pickle.dumps((job_id, result), protocol=pickle.HIGHEST_PROTOCOL)
        view = memoryview(struct.pack(">I", len(payload)) + payload)
        while view:
            view = view[os.write(res_fd, view) :]

    def spawn(job_id, job: dict, timeout: float, mem_bytes: int) -> None:
        workdir = tempfile.mkdtemp(prefix="t2c_job_", dir=tmp_root)
        rfd, wfd = os.pipe()
        pid = os.fork()
        if pid == 0:  # ---- child
            try:
                os.close(rfd)
                os.close(req_fd)
                os.close(res_fd)
                for st in children.values():
                    os.close(st["rfd"])
                signal.signal(signal.SIGINT, signal.SIG_DFL)
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                worker.child_entry(job, wfd, workdir, mem_bytes, int(timeout) + 10, info)
            finally:
                os._exit(1)
        os.close(wfd)
        os.set_blocking(rfd, False)
        children[rfd] = {"id": job_id, "pid": pid, "rfd": rfd, "deadline": time.monotonic() + timeout,
                         "timeout": timeout, "buf": bytearray(), "workdir": workdir}  # fmt: skip
        sel.register(rfd, selectors.EVENT_READ, rfd)

    def reap(st: dict, forced: dict | None = None) -> None:
        sel.unregister(st["rfd"])
        os.close(st["rfd"])
        del children[st["rfd"]]
        try:
            _, status = os.waitpid(st["pid"], 0)
        except ChildProcessError:
            status = 0
        shutil.rmtree(st["workdir"], ignore_errors=True)
        if forced is not None:
            send(st["id"], forced)
            return
        if st["buf"]:
            try:
                send(st["id"], pickle.loads(bytes(st["buf"])))
                return
            except Exception:  # noqa: BLE001 - truncated result: fall through to the crash report
                pass
        if os.WIFSIGNALED(status):
            sig = os.WTERMSIG(status)
            try:
                name = signal.Signals(sig).name
            except ValueError:
                name = f"signal {sig}"
            if name == "SIGXCPU":
                res = {"ok": False, "stage": 1, "error_type": "timeout", "error": "the script exceeded the CPU-time limit"}
            else:
                res = {"ok": False, "stage": 1, "error_type": "crash", "error": f"the CAD kernel crashed ({name}) while running the script"}
        else:
            res = {"ok": False, "stage": 1, "error_type": "crash", "error": f"the worker exited unexpectedly (status {os.WEXITSTATUS(status)})"}
        send(st["id"], res)

    def kill(st: dict) -> None:
        for fn in (os.killpg, os.kill):
            try:
                fn(st["pid"], signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent decides when we stop
    alive = True
    while alive or children:
        now = time.monotonic()
        wait = min((st["deadline"] - now for st in children.values()), default=None)
        for key, _ in sel.select(None if wait is None else max(0.0, min(wait, 1.0))):
            if key.data is None:  # request pipe
                try:
                    chunk = os.read(req_fd, 1 << 20)
                except BlockingIOError:
                    continue
                if not chunk:  # parent closed the pipe (or died): stop everything
                    alive = False
                    sel.unregister(req_fd)
                    for st in list(children.values()):
                        kill(st)
                    continue
                inbuf += chunk
                while len(inbuf) >= 4:
                    (n,) = struct.unpack(">I", inbuf[:4])
                    if len(inbuf) < 4 + n:
                        break
                    job_id, job, timeout, mem_bytes = pickle.loads(bytes(inbuf[4 : 4 + n]))
                    del inbuf[: 4 + n]
                    spawn(job_id, job, timeout, mem_bytes)
            else:
                st = children.get(key.data)
                if st is None:
                    continue
                try:
                    chunk = os.read(st["rfd"], 1 << 20)
                except BlockingIOError:
                    continue
                if chunk:
                    st["buf"] += chunk
                else:
                    reap(st)
        now = time.monotonic()
        for st in [s for s in children.values() if now >= s["deadline"]]:
            kill(st)
            reap(st, {"ok": False, "stage": 1, "error_type": "timeout",
                      "error": f"execution exceeded the {st['timeout']:.0f}-second timeout"})  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
