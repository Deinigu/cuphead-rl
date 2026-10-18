# Removes BepInEx (and the CupheadRLBridge plugin) from the Cuphead folder, restoring it to the
# stock Steam layout. Only removes the root-level files BepInEx added plus the BepInEx\ folder.
param(
    [string]$Game = "E:\SteamLibrary\steamapps\common\Cuphead"
)
$ErrorActionPreference = "Stop"
foreach ($rel in @("winhttp.dll", "doorstop_config.ini", ".doorstop_version", "changelog.txt")) {
    $p = Join-Path $Game $rel
    if (Test-Path $p) { Remove-Item -Force $p; Write-Host "removed $rel" }
}
$bep = Join-Path $Game "BepInEx"
if (Test-Path $bep) { Remove-Item -Recurse -Force $bep; Write-Host "removed BepInEx\" }
Write-Host "Done. Verify with Steam > Cuphead > Properties > Installed Files > Verify integrity."
