# Installs BepInEx 5.4.23.2 (x64) into the Cuphead folder. Adds files only; never overwrites
# original game files (aborts if any target already exists and differs). Writes the list of
# added paths to <game>\BepInEx\.installed_by_cuphead_rl.txt for uninstall_bepinex.ps1.
param(
    [string]$Game = "E:\SteamLibrary\steamapps\common\Cuphead"
)
$ErrorActionPreference = "Stop"
$Url = "https://github.com/BepInEx/BepInEx/releases/download/v5.4.23.2/BepInEx_win_x64_5.4.23.2.zip"
$Sha256 = "F752CE4E838F4C305B9DA1404B6745F2CFF23B8BFD494F79F0C84D0A01F59B46"

$work = Join-Path ([IO.Path]::GetTempPath()) ("bepinex_" + [guid]::NewGuid())
New-Item -ItemType Directory $work | Out-Null
$zip = Join-Path $work "bepinex.zip"
Invoke-WebRequest $Url -OutFile $zip
if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $Sha256) { throw "BepInEx zip hash mismatch" }
$src = Join-Path $work "x"
Expand-Archive $zip -DestinationPath $src

$added = @()
foreach ($f in Get-ChildItem $src -Recurse -File -Force) {
    $rel = $f.FullName.Substring($src.Length + 1)
    $dst = Join-Path $Game $rel
    if (Test-Path $dst) {
        if ((Get-FileHash $dst).Hash -ne (Get-FileHash $f.FullName).Hash) { throw "Refusing to overwrite $dst" }
        continue
    }
    New-Item -ItemType Directory -Force (Split-Path $dst) | Out-Null
    Copy-Item $f.FullName $dst
    $added += $rel
}
# BepInEx creates config/, plugins/, cache, LogOutput.log at first launch; the uninstaller removes
# the whole BepInEx/ folder, so those need not be listed.
$manifest = Join-Path $Game "BepInEx\.installed_by_cuphead_rl.txt"
$added + "BepInEx\.installed_by_cuphead_rl.txt" | Set-Content $manifest
Remove-Item -Recurse -Force $work
Write-Host "Installed BepInEx into $Game ($($added.Count) files). Manifest: $manifest"
