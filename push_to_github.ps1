# push_to_github.ps1
# Pushes this folder into github.com/imanfirdaus27/PortfolioProjects
#
#       cd "C:\Users\firda\Desktop\Master\data-science-portfolio"
#       .\push_to_github.ps1
#
# THIS FOLDER IS THE SOURCE OF TRUTH. If the remote and this folder disagree
# about a file, this folder wins (-X ours). That is deliberate: an earlier
# version of this script used --allow-unrelated-histories on every run, which
# made git treat the two trees as strangers and silently restore old README
# files over the new ones. The flag is now used only on the very first join.

param([string]$RepoUrl = "https://github.com/imanfirdaus27/PortfolioProjects.git")

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$firstTime = -not (Test-Path ".git")
if ($firstTime) { git init | Out-Null }
if (Test-Path ".git\index.lock") { Remove-Item ".git\index.lock" -Force }

git config user.name  "Muhammad Iman Firdaus Bin Md Rostan"
git config user.email "firdausrostan@gmail.com"

if (-not (git remote | Select-String -Quiet "origin")) {
    git remote add origin $RepoUrl
} else {
    git remote set-url origin $RepoUrl
}

if (Test-Path ".git\MERGE_HEAD") {
    Write-Host "Clearing an unfinished merge from a previous run..." -ForegroundColor Yellow
    git merge --abort
}

git add -A
$pending = git status --porcelain
if (-not [string]::IsNullOrWhiteSpace($pending)) {
    git commit -m "Update portfolio"
}

git branch -M main
git fetch origin

$remoteMain = git ls-remote --heads origin main
if (-not [string]::IsNullOrWhiteSpace($remoteMain)) {
    # Do the histories already share a commit?
    $mergeBase = git merge-base HEAD origin/main 2>$null
    if ([string]::IsNullOrWhiteSpace($mergeBase)) {
        Write-Host "First join with the existing repo history..." -ForegroundColor Cyan
        git pull origin main --allow-unrelated-histories --no-rebase --no-edit -X ours
    } else {
        git pull origin main --no-rebase --no-edit -X ours
    }
}

git push -u origin main
Write-Host "`nDone. Open $RepoUrl" -ForegroundColor Green
