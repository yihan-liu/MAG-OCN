# run_test.ps1
# PowerShell script to test spatial segmentation with proper venv activation

param(
    [switch]$SkipInstall,  # Skip dependency installation
    [switch]$Verbose      # Enable verbose output
)

Write-Host "=" * 60 -ForegroundColor Green
Write-Host "MAG-OCN Spatial Segmentation Test Runner" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Green

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# Virtual environment is already activated
Write-Host "✓ Virtual environment already activated" -ForegroundColor Green

# Check Python availability
try {
    $PythonVersion = python --version 2>&1
    Write-Host "✓ Python available: $PythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python and add to PATH." -ForegroundColor Red
    exit 1
}

# Install dependencies if not skipped
if (-not $SkipInstall) {
    if (Test-Path "requirements.txt") {
        Write-Host "Installing/updating dependencies..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ Dependencies installed" -ForegroundColor Green
        } else {
            Write-Host "⚠ Some dependencies may have failed to install" -ForegroundColor Yellow
        }
    } else {
        Write-Host "⚠ requirements.txt not found, skipping dependency installation" -ForegroundColor Yellow
    }
} else {
    Write-Host "Skipping dependency installation (-SkipInstall flag)" -ForegroundColor Yellow
}

# Check for required data files
$DataFile = Join-Path $ScriptDir "raw\1v_expanded.csv"
if (Test-Path $DataFile) {
    Write-Host "✓ Test data file found: $DataFile" -ForegroundColor Green
} else {
    Write-Host "⚠ Test data file not found: $DataFile" -ForegroundColor Yellow
    Write-Host "The test may fail without this file" -ForegroundColor Yellow
}

# Run the test
Write-Host "`nRunning spatial segmentation test..." -ForegroundColor Yellow
Write-Host "Command: python test/test_spatial_segmentation.py" -ForegroundColor Cyan

if ($Verbose) {
    python test/test_spatial_segmentation.py
} else {
    python test/test_spatial_segmentation.py 2>&1
}

$TestResult = $LASTEXITCODE

# Report results
Write-Host "`n" + "=" * 60 -ForegroundColor Green
if ($TestResult -eq 0) {
    Write-Host "✓ TEST COMPLETED SUCCESSFULLY" -ForegroundColor Green
    Write-Host "✓ Spatial segmentation is working correctly" -ForegroundColor Green
    
    # Check for generated files
    if (Test-Path "test/spatial_segmentation_test.png") {
        Write-Host "✓ Visualization generated: test/spatial_segmentation_test.png" -ForegroundColor Green
    }
    
    Write-Host "`nNext Steps:" -ForegroundColor Yellow
    Write-Host "1. Review the test output above" -ForegroundColor White
    Write-Host "2. Run training: python train.py --filenames 1v_expanded --max-atoms-per-segment 80" -ForegroundColor White
    Write-Host "3. Adjust parameters based on your model requirements" -ForegroundColor White
    
} else {
    Write-Host "✗ TEST FAILED" -ForegroundColor Red
    Write-Host "Check the error messages above for troubleshooting" -ForegroundColor Red
    
    Write-Host "`nCommon Issues:" -ForegroundColor Yellow
    Write-Host "1. Missing dependencies - run without -SkipInstall flag" -ForegroundColor White
    Write-Host "2. Missing data files - check ./raw/1v_expanded.csv exists" -ForegroundColor White
    Write-Host "3. Virtual environment issues - try recreating .venv" -ForegroundColor White
}
Write-Host "=" * 60 -ForegroundColor Green

exit $TestResult
