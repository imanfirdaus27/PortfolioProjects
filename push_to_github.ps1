# push_to_github.ps1
# Pushes this folder into the EXISTING repo: github.com/imanfirdaus27/PortfolioProjects
#
# Run from the portfolio folder in PowerShell:
#       cd "C:\Users\firda\Desktop\Master\data-science-portfolio"
#       .\push_to_github.ps1
#
# The remote already has 11 commits of older work. This script merges that history in
# (--allow-unrelated-histories) so nothing there is overwritten, then pushes. The new
# folders (01-..., 02-..., etc.) sit alongside the existing files; no filenames collide.

param([string]$RepoUrl = "https://github.com/imanfirdaus27/PortfolioProjects.git")

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
    git commit -m "Add Master's data science portfolio: 10 projects, code and 65 report figures"
}

git branch -M main

# Bring the existing remote history in before pushing.
git fetch origin
$remoteMain = git ls-remote --heads origin main
if (-not [string]::IsNullOrWhiteSpace($remoteMain)) {
    Write-Host "Merging existing PortfolioProjects history..." -ForegroundColor Cyan
    git pull origin main --allow-unrelated-histories --no-rebase --no-edit
}

git push -u origin main
Write-Host "`nDone. Open https://github.com/imanfirdaus27/PortfolioProjects" -ForegroundColor Green
