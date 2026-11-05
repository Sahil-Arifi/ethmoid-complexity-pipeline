"""Fresh research outputs with an explicit private filesystem access policy."""
from __future__ import annotations

import csv
import io
import os
import re
import subprocess

from . import instance_experiment as legacy


def _windows_acl(path):
    options = {"capture_output": True, "text": True, "timeout": 15,
               "creationflags": 0x08000000}
    identity = subprocess.run(["whoami", "/user", "/fo", "csv", "/nh"], **options)
    rows = list(csv.reader(io.StringIO(identity.stdout)))
    legacy.require(identity.returncode == 0 and len(rows) == 1 and len(rows[0]) == 2,
                   "PROTECTED_OUTPUT_ACL_FAILED")
    sid = rows[0][1].strip()
    legacy.require(re.fullmatch(r"S-\d+(?:-\d+){2,15}", sid) is not None,
                   "PROTECTED_OUTPUT_ACL_FAILED")
    result = subprocess.run(["icacls", str(path), "/inheritance:r", "/grant:r",
                            f"*{sid}:(OI)(CI)F", "*S-1-5-18:(OI)(CI)F"], **options)
    legacy.require(result.returncode == 0, "PROTECTED_OUTPUT_ACL_FAILED")


def new_output(path):
    """Create an external empty directory, then restrict access before any data write.

    Windows grants full access only to the current account and SYSTEM after
    removing inherited grants. POSIX outputs use mode 0700. No command output or
    identity information is emitted; ACL failures return one fixed error code.
    """
    output = legacy.new_output(path)
    try:
        if os.name == "nt":
            _windows_acl(output)
        else:
            output.chmod(0o700)
    except Exception:
        try:
            output.rmdir()  # No caller data has been written into this fresh root.
        except OSError:
            pass
        raise legacy.ExperimentError("PROTECTED_OUTPUT_ACL_FAILED") from None
    return output
