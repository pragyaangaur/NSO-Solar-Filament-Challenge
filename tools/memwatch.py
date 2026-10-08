"""Run a command in its own session and keep its whole process tree under a memory cap.

macOS ignores ulimit memory limits, so this polls the physical footprint that
`top` reports (which includes GPU memory on Apple silicon) for the command and
all of its children. If the total passes the limit, the tree is stopped.
It also detaches the command, so closing the terminal that started it does
not stop it.

Usage: python3 tools/memwatch.py --limit-gb 8 --log outputs/run.log -- ./scripts/run_baseline.sh
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

UNITS = {"B": 1, "K": 2**10, "M": 2**20, "G": 2**30}


def footprints():
    """Return {pid: (ppid, bytes)} for every process, from top's MEM column."""
    out = subprocess.run(["top", "-l", "1", "-stats", "pid,ppid,mem"], capture_output=True, text=True).stdout
    procs = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+(\d+)\s+([\d.]+)([BKMG])[+-]?\s*$", line)
        if m:
            procs[int(m[1])] = (int(m[2]), float(m[3]) * UNITS[m[4]])
    return procs


def tree_bytes(root, procs):
    children = {}
    for pid, (ppid, _) in procs.items():
        children.setdefault(ppid, []).append(pid)
    total, stack = 0.0, [root]
    while stack:
        pid = stack.pop()
        total += procs.get(pid, (0, 0))[1]
        stack.extend(children.get(pid, []))
    return total


def tree_pids(root, procs):
    children = {}
    for pid, (ppid, _) in procs.items():
        children.setdefault(ppid, []).append(pid)
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(children.get(pid, []))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-gb", type=float, default=8)
    ap.add_argument("--log", default="outputs/run.log")
    ap.add_argument("--interval", type=float, default=10)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd

    if os.fork():  # parent returns to the shell; the watcher lives on in its own session
        return
    os.setsid()
    log = open(args.log, "a", buffering=1)
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    limit, peak = args.limit_gb * 2**30, 0.0
    log.write(f"[memwatch] pid {proc.pid}, cap {args.limit_gb} GB\n")
    while proc.poll() is None:
        used = tree_bytes(proc.pid, footprints())
        if used > peak:
            peak = used
            log.write(f"[memwatch] {time.strftime('%H:%M:%S')} new peak {peak / 2**30:.2f} GB\n")
        if used > limit:
            log.write(f"[memwatch] {used / 2**30:.2f} GB is over the cap, stopping the pipeline\n")
            for sig in (signal.SIGTERM, signal.SIGKILL):
                for pid in tree_pids(proc.pid, footprints()):
                    try:
                        os.kill(pid, sig)
                    except ProcessLookupError:
                        pass
                time.sleep(10)
            break
        time.sleep(args.interval)
    log.write(f"[memwatch] finished, exit {proc.wait()}, peak {peak / 2**30:.2f} GB\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
