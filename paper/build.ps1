param()

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $here
try {
  pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Host
  pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Host
}
finally {
  Pop-Location
}
