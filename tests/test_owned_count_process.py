"""Actual process-lifetime checks; no model or medical data is involved."""
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ethmoid_pipeline.webapp.owned_process import launch_owned_process


def test_owned_worker_can_complete_normally():
    child = launch_owned_process([sys.executable, "-c", "raise SystemExit(0)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert child.wait(timeout=10) == 0
    finally:
        child.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows workstation process ownership contract")
def test_hard_server_death_terminates_inference_child(tmp_path):
    ready = tmp_path / "owned-pid.json"
    source_root = Path(__file__).resolve().parents[1] / "src"
    # Killing this stand-in server skips Python finally blocks and atexit hooks.
    parent_code = """
import json, pathlib, subprocess, sys, time
from ethmoid_pipeline.webapp.owned_process import launch_owned_process
child = launch_owned_process([sys.executable, '-c', 'import time; time.sleep(60)'],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid': child.pid}))
time.sleep(60)
"""
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes, api.OpenProcess.restype = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE
    api.WaitForSingleObject.argtypes, api.WaitForSingleObject.restype = [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD
    api.CloseHandle.argtypes, api.CloseHandle.restype = [wintypes.HANDLE], wintypes.BOOL
    parent = subprocess.Popen([sys.executable, "-c", parent_code, str(ready)],
        env={**os.environ, "PYTHONPATH": str(source_root)}, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
    handle = None
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline and parent.poll() is None:
            time.sleep(0.02)
        assert ready.exists(), parent.stderr.read().decode() if parent.poll() is not None else "Owned child did not start"
        pid = json.loads(ready.read_text())["pid"]
        handle = api.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        assert handle
        assert api.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: child is alive
        parent.kill()
        parent.wait(timeout=5)
        assert api.WaitForSingleObject(handle, 5000) == 0  # WAIT_OBJECT_0: kernel terminated child
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        if handle:
            api.CloseHandle(handle)
        if parent.stderr:
            parent.stderr.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object ownership")
def test_job_assignment_failure_cannot_run_child_code(tmp_path, monkeypatch):
    import ethmoid_pipeline.webapp.owned_process as owned
    api = owned._kernel()
    class RejectAssignment:
        def __getattr__(self, name):
            return getattr(api, name)
        def AssignProcessToJobObject(self, *args):
            ctypes.set_last_error(5)
            return False
    monkeypatch.setattr(owned, "_kernel", lambda: RejectAssignment())
    proof = tmp_path / "must-not-exist.txt"
    with pytest.raises(OSError):
        launch_owned_process([sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('ran')", str(proof)],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert not proof.exists()


def test_cancel_requested_during_crash_becomes_terminal_after_recovery(tmp_path):
    from ethmoid_pipeline.webapp.count_queue import CountQueue
    from tests.test_counting_revisions import setup_service
    app, case, service = setup_service(tmp_path)
    # Stop the actual thread; construct only the durable pre-crash rows.
    service.close()
    with app.store.connect() as db:
        source_hash = db.execute("SELECT source_sha256 FROM cases WHERE id=?", (case['case_id'],)).fetchone()[0]
        db.execute("""INSERT INTO count_runs(id,case_id,status,source_sha256,protocol_version,protocol_json,output_relpath,created_at,updated_at)
            VALUES('synthetic_cancel',?,'running',?,'synthetic','{}','synthetic','2026-01-01','2026-01-01')""",
            (case['case_id'], source_hash))
        db.execute("""INSERT INTO count_jobs(run_id,state,cancel_requested,created_at,updated_at)
            VALUES('synthetic_cancel','running',1,'2026-01-01','2026-01-01')""")
    queue = object.__new__(CountQueue)
    queue.store = app.store
    queue._recover()
    with app.store.connect() as db:
        run = db.execute("SELECT status,failure_code FROM count_runs WHERE id='synthetic_cancel'").fetchone()
        job = db.execute("SELECT state,outcome FROM count_jobs WHERE run_id='synthetic_cancel'").fetchone()
    assert tuple(run) == ('failed', 'CANCELLED')
    assert tuple(job) == ('done', 'cancelled')
