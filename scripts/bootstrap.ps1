[CmdletBinding()]
param(
    [switch]$InstallSlicer
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$localRoot = Join-Path $env:LOCALAPPDATA 'EthmoidComplexity'
$dcmRoot = Join-Path $localRoot 'tools\dcm2niix\v1.0.20260724'
$dcmZip = Join-Path $dcmRoot 'dcm2niix_win.zip'
$dcmBin = Join-Path $dcmRoot 'bin'
$slicerRoot = Join-Path $localRoot 'tools\slicer\5.12.3-r34627'
$slicerInstaller = Join-Path $slicerRoot 'Slicer-5.12.3-win-amd64.exe'
$slicerApp = Join-Path $slicerRoot 'app'

New-Item -ItemType Directory -Force -Path $dcmRoot, $slicerRoot, (Join-Path $localRoot 'work') | Out-Null

if (-not (Test-Path -LiteralPath $dcmZip)) {
    Invoke-WebRequest -Uri 'https://github.com/rordenlab/dcm2niix/releases/download/v1.0.20260724/dcm2niix_win.zip' -OutFile $dcmZip
}
$dcmHash = (Get-FileHash -LiteralPath $dcmZip -Algorithm SHA256).Hash.ToLowerInvariant()
if ($dcmHash -ne '8310549ef7340b1b169c26f4721aeac69423fd73643fbbd22eec47b36cc70ada') {
    throw 'dcm2niix archive checksum mismatch.'
}
$expectedDcmFiles = [ordered]@{
    'concrt140.dll' = '2405355f0a58067b258f8df33c327e3a3d716eaac5a3a5aebb757842d85bd376'
    'dcm2niix.exe' = 'de2466c1e31bb0114ad2c5a3fdfa08be62447b25beb4f4f6a426c314797e48d8'
    'msvcp140.dll' = '0f885b509a685d2bbfa652fed26b5fb31d88fbdab0a978c641d1c7b8aa460aa9'
    'msvcp140_1.dll' = 'bfad5aef4c63a669e3c140655cdfdf395b6c979b400a447bd5dcb65ed8826c3d'
    'msvcp140_2.dll' = '3ea06f0ee098b4823cb79599df3780e7f23cce52c19aac31d2a0d47efe33a5e9'
    'msvcp140_atomic_wait.dll' = '640b2aefced484d0368eea5bdd06addd0658a3a70a49256e560d6923b404a479'
    'msvcp140_codecvt_ids.dll' = 'f2069a52880ec885ee7f0511186100eb7fada0411a2b4948fafea7735b878a18'
    'vcruntime140.dll' = 'd5e4d9a3e835fa679450145d6a7d94e36573a509317111904d9b3712c30d9066'
    'vcruntime140_1.dll' = '1f2d41c4aa5db0bc33ebf7b66d72943a817d7ce6cbe880502a9403823633093f'
}

function Test-DcmBin([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) { return $false }
    $actual = @(Get-ChildItem -LiteralPath $Path -File)
    if ($actual.Count -ne $expectedDcmFiles.Count) { return $false }
    foreach ($entry in $expectedDcmFiles.GetEnumerator()) {
        $file = Join-Path $Path $entry.Key
        if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { return $false }
        $hash = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -ne $entry.Value) { return $false }
    }
    return $true
}

if (-not (Test-DcmBin $dcmBin)) {
    if (Test-Path -LiteralPath $dcmBin) {
        throw 'Existing dcm2niix bin is not the exact pinned file set; remove it manually before bootstrap.'
    }
    $staging = Join-Path $dcmRoot ('.extract-' + [guid]::NewGuid().ToString('N'))
    $resolvedRoot = [IO.Path]::GetFullPath($dcmRoot)
    $resolvedStaging = [IO.Path]::GetFullPath($staging)
    if (-not $resolvedStaging.StartsWith($resolvedRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Unsafe dcm2niix staging path.'
    }
    try {
        New-Item -ItemType Directory -Path $staging | Out-Null
        Expand-Archive -LiteralPath $dcmZip -DestinationPath $staging
        if (-not (Test-DcmBin $staging)) { throw 'Extracted dcm2niix file manifest mismatch.' }
        Move-Item -LiteralPath $staging -Destination $dcmBin
    }
    finally {
        if (Test-Path -LiteralPath $staging) {
            Remove-Item -LiteralPath $staging -Recurse -Force
        }
    }
}

if ($InstallSlicer) {
    if (-not (Test-Path -LiteralPath $slicerInstaller)) {
        Invoke-WebRequest -Uri 'https://download.slicer.org/bitstream/6a61770a2eb3d967f03268b4' -OutFile $slicerInstaller
    }
    $slicerHash = (Get-FileHash -LiteralPath $slicerInstaller -Algorithm SHA512).Hash.ToLowerInvariant()
    if ($slicerHash -ne '109508d54b204f75371f6ae99b9951993a6b753a91c855cd223fcb8eeb65b37bac4f1096d13c1cae89b4d525910b238239a5fb05c98d49874ff090315cd19315') {
        throw '3D Slicer installer checksum mismatch.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $slicerApp 'Slicer.exe'))) {
        $process = Start-Process -FilePath $slicerInstaller -ArgumentList '/S', "/D=$slicerApp" -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0) { throw "3D Slicer installer exited $($process.ExitCode)." }
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $repo '.venv\Scripts\python.exe'))) {
    & py -3.13 -c "import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 13, 12) else 1)"
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13.12 is required by the pinned pilot environment.' }
    py -3.13 -m venv (Join-Path $repo '.venv')
}
$python = Join-Path $repo '.venv\Scripts\python.exe'
& $python -c "import sys; raise SystemExit(0 if sys.version_info[:3] == (3, 13, 12) else 1)"
if ($LASTEXITCODE -ne 0) { throw 'The project virtual environment must use Python 3.13.12.' }
& $python -m pip install --disable-pip-version-check --require-hashes -r (Join-Path $repo 'requirements.lock')
& $python -m pip install --disable-pip-version-check --no-build-isolation --no-deps -e $repo

$nodeVersion = (& node --version 2>$null)
if ($LASTEXITCODE -ne 0 -or $nodeVersion -ne 'v24.14.1') {
    throw 'Node.js 24.14.1 is required by the pinned web application environment.'
}
$npmVersion = (& npm --version 2>$null)
if ($LASTEXITCODE -ne 0 -or $npmVersion -ne '11.11.0') {
    throw 'npm 11.11.0 is required by the pinned web application environment.'
}
$webRoot = Join-Path $repo 'web'
Push-Location $webRoot
try {
    & npm ci
    if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    & npm run build
    if ($LASTEXITCODE -ne 0) { throw 'The web application build failed.' }
}
finally {
    Pop-Location
}

Write-Host 'Bootstrap complete. Start the local application with: ethmoid-app start'
