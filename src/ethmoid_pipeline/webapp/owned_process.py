"""Own Windows inference descendants across abrupt server death.

Windows children start suspended and join a private kill-on-close Job Object
before their initial thread resumes. Inference therefore cannot outlive the
server's ownership handle or run before assignment succeeds. No handle is
inherited by the worker. Non-Windows test hosts use an isolated process group.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import signal
import subprocess


class _BasicLimit(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in ("ReadOperationCount", "WriteOperationCount",
        "OtherOperationCount", "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimit), ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ThreadID", wintypes.DWORD),
        ("th32OwnerProcessID", wintypes.DWORD), ("tpBasePri", wintypes.LONG),
        ("tpDeltaPri", wintypes.LONG), ("dwFlags", wintypes.DWORD)]


def _kernel():
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        "SetInformationJobObject": ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        "AssignProcessToJobObject": ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        "CloseHandle": ([wintypes.HANDLE], wintypes.BOOL),
        "CreateToolhelp32Snapshot": ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        "Thread32First": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        "Thread32Next": ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        "OpenThread": ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        "ResumeThread": ([wintypes.HANDLE], wintypes.DWORD),
    }
    for name, (args, result) in signatures.items():
        operation = getattr(api, name)
        operation.argtypes, operation.restype = args, result
    return api


def _resume_initial_thread(api, pid):
    snapshot = api.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        available = api.Thread32First(snapshot, ctypes.byref(entry))
        while available:
            if entry.th32OwnerProcessID == pid:
                thread = api.OpenThread(0x0002, False, entry.th32ThreadID)  # THREAD_SUSPEND_RESUME
                if not thread:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if api.ResumeThread(thread) == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    return
                finally:
                    api.CloseHandle(thread)
            available = api.Thread32Next(snapshot, ctypes.byref(entry))
        raise OSError("The suspended inference process has no initial thread")
    finally:
        api.CloseHandle(snapshot)


class OwnedProcess:
    def __init__(self, process, *, job=None, api=None):
        self.process, self._job, self._api = process, job, api

    @property
    def pid(self):
        return self.process.pid

    @property
    def returncode(self):
        return self.process.returncode

    def poll(self):
        return self.process.poll()

    def wait(self, timeout=None):
        return self.process.wait(timeout=timeout)

    def terminate(self):
        if self._job is not None:
            self.close()
        elif self.process.poll() is None:
            if os.name == "nt":
                self.process.terminate()
            else:
                os.killpg(self.process.pid, signal.SIGTERM)

    def close(self):
        if self._job is not None:
            self._api.CloseHandle(self._job)
            self._job = None
        elif os.name != "nt" and self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGKILL)


def launch_owned_process(command, **kwargs) -> OwnedProcess:
    """Start an inference process whose Windows descendants die with its owner."""
    if os.name != "nt":
        return OwnedProcess(subprocess.Popen(command, start_new_session=True, **kwargs))
    api = _kernel()
    job = api.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    process = None
    try:
        limits = _ExtendedLimit()
        limits.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not api.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        options = dict(kwargs)
        options["creationflags"] = options.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW | 0x00000004  # CREATE_SUSPENDED
        options["close_fds"] = True
        process = subprocess.Popen(command, **options)
        if not api.AssignProcessToJobObject(job, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())
        _resume_initial_thread(api, process.pid)
        return OwnedProcess(process, job=job, api=api)
    except BaseException:
        if process is not None:
            process.kill()
            process.wait(timeout=10)
        api.CloseHandle(job)
        raise
