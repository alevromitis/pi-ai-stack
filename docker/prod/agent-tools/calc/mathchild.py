"""The child process side of symbolic_math and physics_units (server.py).

server.py imports only this small module; the forkserver preloads
mathtools (sympy, pint) and forks one child per call, which server.py kills
when it runs past its time limit. A child the kernel must kill for memory
should be the child, not the server: it asks for that first.
"""

from __future__ import annotations

import resource


def run(conn, name: str, args: tuple, cpu_seconds: int) -> None:
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("1000")
    except OSError:
        pass
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
    import mathtools

    try:
        conn.send(("ok", getattr(mathtools, name)(*args)))
    except mathtools.MathError as exc:
        conn.send(("error", str(exc)))
    except Exception as exc:  # sympy and pint raise many kinds; say which, not the trace
        conn.send(("error", f"failed ({type(exc).__name__})"))
    finally:
        conn.close()


def ping(conn) -> None:
    conn.send("ok")
    conn.close()
