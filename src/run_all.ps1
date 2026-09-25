param(
    [switch]$RandomNetworkOnly,
    [switch]$RealWorldOnly
)

$ErrorActionPreference = "Stop"

if ($RandomNetworkOnly -and $RealWorldOnly) {
    throw "Specify at most one of -RandomNetworkOnly and -RealWorldOnly."
}

$RunRandomNetwork = -not $RealWorldOnly
$RunRealWorld = -not $RandomNetworkOnly

$SrcRoot = $PSScriptRoot
$CppRoot = Join-Path $SrcRoot "cpp"
$BuildRoot = Join-Path $CppRoot "build"
$BinRoot = Join-Path $BuildRoot "bin\Release"
$PythonRoot = Join-Path $SrcRoot "python"

$RandomExecutable = Join-Path $BinRoot "random_graph_classification.exe"
$RealWorldExecutable = Join-Path $BinRoot "real_world_data.exe"

$RawRoot = Join-Path $CppRoot "results\raw"
$RandomRawRoot = Join-Path $RawRoot "random_graph_classification"
$RealWorldRawRoot = Join-Path $RawRoot "real_world_data"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )

    & $Executable @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw "'$Executable' failed with exit code $LASTEXITCODE."
    }
}

function Assert-FileExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Description was not produced: '$Path'."
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "1. Cleaning C++ build"
Write-Host "============================================================"

if (Test-Path -LiteralPath $BuildRoot) {
    Remove-Item `
        -LiteralPath $BuildRoot `
        -Recurse `
        -Force
}

Write-Host ""
Write-Host "============================================================"
Write-Host "2. Configuring C++"
Write-Host "============================================================"

Invoke-Checked `
    cmake `
    -S $CppRoot `
    -B $BuildRoot

Write-Host ""
Write-Host "============================================================"
Write-Host "3. Building selected C++ Release executables"
Write-Host "============================================================"

$BuildTargets = @()

if ($RunRandomNetwork) {
    $BuildTargets += "random_graph_classification"
}

if ($RunRealWorld) {
    $BuildTargets += "real_world_data"
}

Invoke-Checked `
    cmake `
    --build $BuildRoot `
    --config Release `
    --target @BuildTargets

if ($RunRandomNetwork) {
    Assert-FileExists `
        $RandomExecutable `
        "The random-network executable"
}

if ($RunRealWorld) {
    Assert-FileExists `
        $RealWorldExecutable `
        "The real-world executable"
}

if ($RunRandomNetwork) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "4. Running random-network C++ experiments"
    Write-Host "============================================================"
    Write-Host "Output root: $RandomRawRoot"

    Invoke-Checked `
        $RandomExecutable `
        --root $SrcRoot `
        --output-root $RandomRawRoot `
        --force

    Write-Host ""
    Write-Host "Verifying random-network C++ output..."

    $RandomConfiguration = Join-Path $RandomRawRoot "configuration.json"

    Assert-FileExists `
        $RandomConfiguration `
        "Random-network configuration.json"

    Write-Host "Random-network raw output verified."
}

if ($RunRealWorld) {
    Write-Host ""
    Write-Host "============================================================"
    Write-Host "5. Clearing real-world C++ raw cache"
    Write-Host "============================================================"

    if (Test-Path -LiteralPath $RealWorldRawRoot) {
        Remove-Item `
            -LiteralPath $RealWorldRawRoot `
            -Recurse `
            -Force
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "6. Running real-world C++ cache generation"
    Write-Host "============================================================"
    Write-Host "Output root: $RealWorldRawRoot"

    Invoke-Checked `
        $RealWorldExecutable `
        --root $SrcRoot

    Write-Host ""
    Write-Host "Verifying real-world C++ output..."

    $RealWorldDatasets = Join-Path $RealWorldRawRoot "datasets.csv"
    $RealWorldGraphs = Join-Path $RealWorldRawRoot "graphs.csv"
    $RealWorldPersistence = Join-Path $RealWorldRawRoot "persistence_diagrams.csv"
    $RealWorldHarmonic = Join-Path $RealWorldRawRoot "harmonic_features.csv"
    $RealWorldCharacters = Join-Path $RealWorldRawRoot "characters.csv"

    Assert-FileExists `
        $RealWorldDatasets `
        "Real-world datasets.csv"

    Assert-FileExists `
        $RealWorldGraphs `
        "Real-world graphs.csv"

    Assert-FileExists `
        $RealWorldPersistence `
        "Real-world persistence_diagrams.csv"

    Assert-FileExists `
        $RealWorldHarmonic `
        "Real-world harmonic_features.csv"

    Assert-FileExists `
        $RealWorldCharacters `
        "Real-world characters.csv"

    Write-Host "Real-world raw output verified."
}

Push-Location $PythonRoot

try {
    if ($RunRandomNetwork) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "7. Running random-network Python analysis"
        Write-Host "============================================================"

        Invoke-Checked `
            python `
            -m experiments.random_graph_classification.main
    }

    if ($RunRealWorld) {
        Write-Host ""
        Write-Host "============================================================"
        Write-Host "8. Running real-world Python analysis"
        Write-Host "============================================================"

        Invoke-Checked `
            python `
            -m experiments.real_world_data.main
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "============================================================"

if ($RandomNetworkOnly) {
    Write-Host "RANDOM-NETWORK EXPERIMENT COMPLETE"
}
elseif ($RealWorldOnly) {
    Write-Host "REAL-WORLD EXPERIMENT COMPLETE"
}
else {
    Write-Host "ALL EXPERIMENTS COMPLETE"
}

Write-Host "============================================================"