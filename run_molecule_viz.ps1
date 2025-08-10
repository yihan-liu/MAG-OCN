# run_molecule_viz.ps1
# PowerShell script to run molecular visualization

param(
    [string]$Molecule = "0v",
    [string]$Checkpoint = "checkpoints/best_checkpoint.pt",
    [string]$Output = $null,
    [int]$MaxAtoms = 50,
    [switch]$NoArrows,
    [switch]$NoPredict,
    [string]$Device = "auto"
)

Write-Host "🧬 MAG-OCN Molecular Visualization Runner" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Activate Python environment if needed
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
    & .\venv\Scripts\Activate.ps1
}

# Build command arguments
$Args = @(
    "molecule_prediction_viz.py",
    "--molecule", $Molecule,
    "--checkpoint", $Checkpoint,
    "--max-atoms-per-segment", $MaxAtoms,
    "--device", $Device
)

if ($Output) {
    $Args += "--output", $Output
}

if ($NoArrows) {
    $Args += "--no-arrows"
}

if ($NoPredict) {
    $Args += "--no-prediction"
}

Write-Host "🚀 Running molecular visualization..." -ForegroundColor Green
Write-Host "   Molecule: $Molecule" -ForegroundColor White
Write-Host "   Checkpoint: $Checkpoint" -ForegroundColor White
Write-Host "   Max atoms per segment: $MaxAtoms" -ForegroundColor White
Write-Host "   Device: $Device" -ForegroundColor White

# Run the visualization
python @Args

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Visualization completed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Visualization failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}

# Examples:
# .\run_molecule_viz.ps1 -Molecule "0v" -NoPredict                    # Structure only
# .\run_molecule_viz.ps1 -Molecule "1v" -MaxAtoms 30                  # With predictions, smaller segments
# .\run_molecule_viz.ps1 -Molecule "raw/2v.csv" -Output "my_viz.png"  # Custom output path
