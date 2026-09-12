# push_to_github.ps1
# 1. Create an EMPTY public repo on github.com named: data-science-portfolio
#    (no README, no .gitignore, no licence - this folder already has them)
# 2. Run this from the portfolio folder in PowerShell:
#       cd "C:\Users\firda\Desktop\Master\data-science-portfolio"
#       .\push_to_github.ps1

param([string]$RepoUrl = "https://github.com/imanfirdaus27/data-science-portfolio.git")

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".git")) { git init | Out-Null }
if (Test-Path ".git\index.lock") { Remove-Item ".git\index.lock" -Force }

git config user.name  "Muhammad Iman Firdaus Bin Md Rostan"
git config user.email "firdausrostan@gmail.com"

if (-not (git remote | Select-String -Quiet "origin")) {
    git remote add origin $RepoUrl
} else {
    git remote set-url origin $RepoUrl
}

git add -A
$pending = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($pending)) {
    git commit -m "Update portfolio"
}

git branch -M main
git push -u origin main
Write-Host "`nDone. Open $RepoUrl" -ForegroundColor Green
