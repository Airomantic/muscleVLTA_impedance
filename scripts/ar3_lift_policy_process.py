#!/usr/bin/env python3
"""Exec an owned policy command that Linux kills if its debug session disappears.

Run this small, separate process before importing model libraries. Setting prctl in
Popen's preexec_fn would be unsafe in the debugger's multithreaded process.
"""

import argparse
import ctypes
import os
import signal
import sys


def bind_to_parent(expected_pid: int) -> None:
    if sys.platform != "linux":
        raise RuntimeError("owned policy processes require Linux parent-death signals")
    if os.getppid() != expected_pid:
        raise RuntimeError("debug owner process exited before policy startup")
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    # PR_SET_PDEATHSIG survives normal exec, including bash -> env -> model Python.
    # SIGKILL also works during a long native model import or GPU compilation.
    if prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # The parent may have died between the first check and prctl registration.
    if os.getppid() != expected_pid:
        raise RuntimeError("debug owner process exited during policy startup")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if args.parent_pid <= 0 or not command:
        parser.error("a positive --parent-pid and a command after -- are required")
    try:
        bind_to_parent(args.parent_pid)
        os.execvp(command[0], command)
    except (OSError, RuntimeError) as error:
        print(f"[ar3_lift_policy_process] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
