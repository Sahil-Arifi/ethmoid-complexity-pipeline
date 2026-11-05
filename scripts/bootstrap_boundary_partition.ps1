param(
    [string]$RuntimeRoot = (Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\tools\boundary-partition-runtime'),
    [switch]$RefreshSeal
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceLockPath = Join-Path $repoRoot 'boundary-partition-source.lock.json'
$packageLockPath = Join-Path $repoRoot 'boundary-partition-conda.lock.json'
$explicitLockPath = Join-Path $repoRoot 'boundary-partition-win-64.explicit.txt'
$sourceLock = Get-Content -LiteralPath $sourceLockPath -Raw | ConvertFrom-Json
$resolvedRoot = [IO.Path]::GetFullPath($RuntimeRoot)
$requiredRoot = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'EthmoidComplexity\tools'))
if (-not $resolvedRoot.StartsWith($requiredRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'PARTITION_EXTERNAL_TOOLS_ROOT_REQUIRED'
}
New-Item -ItemType Directory -Path $resolvedRoot -Force | Out-Null
# Only the current Windows account and SYSTEM may access retained job arrays.
$identity = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$acl = Get-Acl -LiteralPath $resolvedRoot
$allowed = @($identity, 'NT AUTHORITY\SYSTEM')
$unexpected = @($acl.Access | Where-Object { $_.AccessControlType -eq 'Allow' -and $_.IdentityReference.Value -notin $allowed })
if (-not $acl.AreAccessRulesProtected -or $unexpected.Count -gt 0) {
    & icacls.exe $resolvedRoot /inheritance:r /grant:r "${identity}:(OI)(CI)F" 'SYSTEM:(OI)(CI)F' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'PARTITION_PRIVATE_ACL_FAILED' }
}

function Get-VerifiedDownload([string]$Url, [string]$Destination, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Destination)) {
        Invoke-WebRequest -Uri $Url -OutFile $Destination
    }
    if ((Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {
        throw 'PARTITION_DOWNLOAD_CHECKSUM_MISMATCH'
    }
}
$micromamba = Join-Path $resolvedRoot 'micromamba.exe'
Get-VerifiedDownload $sourceLock.micromamba_url $micromamba $sourceLock.micromamba_sha256
$prefix = Join-Path $resolvedRoot 'env'
$python = Join-Path $prefix 'python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & $micromamba --no-rc create --yes --prefix $prefix --root-prefix (Join-Path $resolvedRoot 'mamba') --file $explicitLockPath
    if ($LASTEXITCODE -ne 0) { throw 'PARTITION_ENVIRONMENT_INSTALL_FAILED' }
}
$archive = Join-Path $resolvedRoot ('panseg-' + $sourceLock.panseg_commit + '.zip')
Get-VerifiedDownload $sourceLock.panseg_archive_url $archive $sourceLock.panseg_archive_sha256
$sourceDirectory = Join-Path $resolvedRoot 'source'
$sourceRoot = Join-Path $sourceDirectory ('panseg-' + $sourceLock.panseg_commit)
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    Expand-Archive -LiteralPath $archive -DestinationPath $sourceDirectory
}
$priorPythonPath = $env:PYTHONPATH
$priorPath = $env:PATH
$priorNoBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = (Join-Path $repoRoot 'src') + ';' + $sourceRoot
    $env:PATH = (Join-Path $prefix 'Library\bin') + ';' + $priorPath
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $manifest = Join-Path $resolvedRoot 'runtime-manifest.local.json'
    if ((Test-Path -LiteralPath $manifest) -and $RefreshSeal) {
        $suffix = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffff')
        Move-Item -LiteralPath $manifest -Destination (Join-Path $resolvedRoot ("runtime-manifest-$suffix.local.json"))
    }
    if (-not (Test-Path -LiteralPath $manifest)) {
        & $python -s -m ethmoid_pipeline.counting.boundary_partition --seal-runtime $resolvedRoot --conda-lock $packageLockPath
        if ($LASTEXITCODE -ne 0) { throw 'PARTITION_RUNTIME_SEAL_FAILED' }
    }
    & $python -s -m ethmoid_pipeline.counting.boundary_partition --verify-runtime $manifest
    if ($LASTEXITCODE -ne 0) { throw 'PARTITION_RUNTIME_VERIFY_FAILED' }
    Write-Output $manifest
} finally {
    $env:PYTHONPATH = $priorPythonPath
    $env:PATH = $priorPath
    $env:PYTHONDONTWRITEBYTECODE = $priorNoBytecode
}
