---
tags: [concept]
aliases: [sandbox, fork server, zygote, namespaces, rlimits]
---
# Sandboxing untrusted code

Model-written code is *untrusted*: not malicious, but capable of infinite loops, 40 GB
allocations, segfaults in a C++ library, deleting files or opening connections. Running
it inside the training process would let one bad answer kill a 10-hour job. The
techniques used in [[cad.sandbox]], and what to read about each:

| Technique | Protects against | Here |
|---|---|---|
| **one process per script** | crashes (segfaults) | each script runs in a forked child |
| **fork server / "zygote"** | slow start-up | one process imports cadquery once, then `fork()`s copies (copy-on-write memory, ~10 ms) |
| **process groups + `SIGKILL`** | infinite loops, stray child processes | the whole group is killed at the deadline |
| **resource limits (`setrlimit`)** | memory blow-ups, huge files, CPU spinning, core dumps | per child |
| **Linux namespaces** | network access | an unprivileged *user* namespace allows an empty *network* namespace: no interfaces, no connections |
| **static analysis (`ast`)** | obvious accidents (`import subprocess`, `os.remove`) | refused before running |
| **scratch directory** | clutter | each child runs in its own temporary directory, deleted afterwards |

What it does **not** do: hide your files (a script can read and write what your user
can). Stronger isolation would use containers (e.g. Apptainer on HPC systems), seccomp
filters or gVisor — overkill for model-written CAD scripts, necessary for adversarial code.

## Learn more
- Linux man pages: namespaces(7) <https://man7.org/linux/man-pages/man7/namespaces.7.html> ·
  user_namespaces(7) <https://man7.org/linux/man-pages/man7/user_namespaces.7.html> ·
  network_namespaces(7) <https://man7.org/linux/man-pages/man7/network_namespaces.7.html> ·
  setrlimit(2) <https://man7.org/linux/man-pages/man2/setrlimit.2.html> · fork(2)
  <https://man7.org/linux/man-pages/man2/fork.2.html>
- Python `multiprocessing` start methods (fork, spawn, forkserver — the same idea as our
  zygote): <https://docs.python.org/3/library/multiprocessing.html>
- Python `ast` module: <https://docs.python.org/3/library/ast.html>
