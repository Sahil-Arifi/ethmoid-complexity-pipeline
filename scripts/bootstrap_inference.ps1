[CmdletBinding()]
param(
    [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\tools\counting-runtime'),
    [switch]$RefreshLock,
    [switch]$SkipWeights
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$allowedTools = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\tools'))
$RuntimeRoot = [IO.Path]::GetFullPath($RuntimeRoot)
if (-not $RuntimeRoot.StartsWith($allowedTools + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Inference runtime must be inside the protected local tools directory.'
}
$python = Join-Path $RuntimeRoot 'Scripts\python.exe'
$lock = Join-Path $projectRoot 'inference-requirements.lock'
$requirements = Join-Path $projectRoot 'inference-requirements.in'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw 'Install uv before bootstrapping inference.' }
if (-not (Test-Path -LiteralPath $python)) {
    New-Item -ItemType Directory -Force -Path $allowedTools | Out-Null
    Push-Location $allowedTools
    try { uv venv --python 3.12.14 $RuntimeRoot }
    finally { Pop-Location }
    if ($LASTEXITCODE -ne 0) { throw 'Inference environment creation failed.' }
}
$pythonVersion = & $python -c 'import platform; print(platform.python_version())'
if ($pythonVersion.Trim() -ne '3.12.14') { throw 'Existing inference runtime must use Python 3.12.14.' }
if ($RefreshLock) {
    uv pip compile $requirements --python $python --python-platform windows --torch-backend cu130 --generate-hashes --output-file $lock --custom-compile-command 'uv pip compile inference-requirements.in --python-version 3.12 --python-platform windows --torch-backend cu130 --generate-hashes -o inference-requirements.lock' --quiet
    if ($LASTEXITCODE -ne 0) { throw 'Inference lock resolution failed.' }
}
if (-not (Test-Path -LiteralPath $lock)) { throw 'The committed inference lock is missing.' }
uv pip sync $lock --python $python --torch-backend cu130 --require-hashes --quiet
if ($LASTEXITCODE -ne 0) { throw 'Inference dependency installation failed.' }
uv pip check --python $python
if ($LASTEXITCODE -ne 0) { throw 'Inference dependency consistency check failed.' }

$env:ETHMOID_INFERENCE_RUNTIME = $RuntimeRoot
$env:TOTALSEG_HOME_DIR = Join-Path $RuntimeRoot 'totalsegmentator'
$env:TOTALSEG_WEIGHTS_PATH = Join-Path $env:TOTALSEG_HOME_DIR 'nnunet\results'
$env:nnUNet_compile = 'false'
New-Item -ItemType Directory -Force -Path $env:TOTALSEG_WEIGHTS_PATH | Out-Null

# Installed into this isolated environment only. Spawned Python workers inherit
# the same default-on policy. Loopback is needed by multiprocessing managers.
$offlineGuard = @'
"""Local-only Python network guard installed by bootstrap_inference.ps1."""
import ipaddress
import os
from pathlib import Path
import sys

_runtime = Path(sys.prefix)
os.environ.setdefault("TOTALSEG_HOME_DIR", str(_runtime / "totalsegmentator"))
os.environ.setdefault("TOTALSEG_WEIGHTS_PATH", str(_runtime / "totalsegmentator" / "nnunet" / "results"))
os.environ.setdefault("nnUNet_compile", "false")

def _local_host(host):
    if host is None or host == "":
        return True
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="strict")
    if str(host).lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%", 1)[0]).is_loopback
    except ValueError:
        return False

def _offline_audit(event, arguments):
    if event == "socket.getaddrinfo" and not _local_host(arguments[0]):
        raise PermissionError("External DNS is disabled in the inference runtime.")
    if event in {"socket.connect", "socket.sendto"}:
        address = arguments[-1]
        if isinstance(address, tuple) and not _local_host(address[0]):
            raise PermissionError("External network access is disabled in the inference runtime.")

if os.environ.get("ETHMOID_INFERENCE_OFFLINE", "1") != "0":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.addaudithook(_offline_audit)
'@
[IO.File]::WriteAllText((Join-Path $RuntimeRoot 'Lib\site-packages\sitecustomize.py'), $offlineGuard + "`n", [Text.UTF8Encoding]::new($false))

# Set privacy configuration BEFORE importing the TotalSegmentator API or giving
# any process a patient image. These are model downloads, never scan uploads.
@'
from pathlib import Path
import json, os
root = Path(os.environ["TOTALSEG_HOME_DIR"])
path = root / "config.json"
config = json.loads(path.read_text()) if path.exists() else {}
config.update(send_usage_stats=False, statistics_disclaimer_shown=True)
config.setdefault("totalseg_id", "ethmoid-local-offline")
config.setdefault("prediction_counter", 0)
path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
assert json.loads(path.read_text())["send_usage_stats"] is False
print("TotalSegmentator usage telemetry disabled.")
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'Telemetry configuration failed.' }

if (-not $SkipWeights) {
    $previousOffline = $env:ETHMOID_INFERENCE_OFFLINE
    try {
        $env:ETHMOID_INFERENCE_OFFLINE = '0'
        @'
from pathlib import Path
import contextlib, hashlib, importlib.metadata, io, json, os
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.map_tasks_config import TASK_CONFIGS, TASK_ID_WEIGHTS_CONFIGS

root = Path(os.environ["ETHMOID_INFERENCE_RUNTIME"])
weights = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
assert json.loads((Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json").read_text())["send_usage_stats"] is False
assert TASK_CONFIGS["craniofacial_structures"]["task_id"] == 115
assert TASK_CONFIGS["head_glands_cavities"]["task_id"] == 775
models = []
for task_id in (115, 775, 297, 298):
    info = TASK_ID_WEIGHTS_CONFIGS[task_id]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        download_pretrained_weights(task_id)
    directory = weights / info["foldername"]
    if not directory.is_dir() or not list(directory.rglob("checkpoint_final.pth")):
        raise RuntimeError("A required model download is incomplete.")
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"path": path.relative_to(weights).as_posix(), "sha256": digest, "bytes": path.stat().st_size})
    models.append({"task_id": task_id, "folder": info["foldername"], "url": "https://github.com/wasserth/TotalSegmentator/releases/download/" + info["version"] + "/" + info["foldername"] + ".zip", "files": files})
    print("Verified local model " + str(task_id) + ".", flush=True)
manifest = {"schema_version": 1, "totalsegmentator_version": importlib.metadata.version("TotalSegmentator"), "telemetry_enabled": False, "models": models}
path = root / "weights-manifest.json"
if path.exists() and json.loads(path.read_text()) != manifest:
    raise RuntimeError("Existing model manifest differs; investigate before accepting changed weights.")
path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print("Verified model manifests: tasks 115, 775, 297, 298.")
'@ | & $python -
        if ($LASTEXITCODE -ne 0) { throw 'Model prefetch or hash verification failed.' }
    }
    finally { $env:ETHMOID_INFERENCE_OFFLINE = $previousOffline }
}

$env:ETHMOID_INFERENCE_OFFLINE = '1'
@'
from pathlib import Path
import importlib.metadata, json, os, platform, socket, sys
import torch

assert torch.cuda.is_available(), "CUDA is unavailable."
assert "sm_120" in torch.cuda.get_arch_list(), "This PyTorch build lacks Blackwell support."
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
a = torch.randn(64, 64)
b = torch.randn(64, 64)
actual = (a.cuda() @ b.cuda()).cpu()
torch.testing.assert_close(actual, a @ b, rtol=1e-4, atol=1e-4)
conv = torch.nn.Conv3d(1, 2, 3, padding=1).eval()
x = torch.randn(1, 1, 16, 16, 16)
with torch.inference_mode():
    expected = conv(x)
    observed = conv.cuda()(x.cuda()).cpu()
torch.testing.assert_close(observed, expected, rtol=1e-4, atol=1e-4)
torch.cuda.synchronize()
blocked = False
try:
    socket.getaddrinfo("example.invalid", 443)
except PermissionError:
    blocked = True
assert blocked, "Offline network guard is inactive."
assert json.loads((Path(os.environ["TOTALSEG_HOME_DIR"]) / "config.json").read_text())["send_usage_stats"] is False
report = {"schema_version": 1, "status": "PASS", "python": platform.python_version(), "platform": platform.system(), "gpu": torch.cuda.get_device_name(0), "gpu_capability": list(torch.cuda.get_device_capability(0)), "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory, "torch_cuda": torch.version.cuda, "packages": {p: importlib.metadata.version(p) for p in ("torch", "TotalSegmentator", "nnunetv2", "numpy", "scipy", "scikit-image", "nibabel", "pytest")}, "checks": ["CUDA matrix multiplication matches CPU", "CUDA 3D convolution matches CPU", "external DNS blocked", "usage telemetry disabled"]}
(Path(os.environ["ETHMOID_INFERENCE_RUNTIME"]) / "runtime-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
'@ | & $python -
if ($LASTEXITCODE -ne 0) { throw 'GPU or offline runtime smoke test failed.' }
Write-Output 'Inference runtime ready. Patient images remain local; models provide anatomical landmarks only.'
