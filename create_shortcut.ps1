<#
  Creates a "Jadu" shortcut on your Desktop and in the Start menu (so you can search "Jadu"
  or pin it to the taskbar). Run once from the project folder:

      powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1
      powershell -ExecutionPolicy Bypass -File .\create_shortcut.ps1 -Minimized   # start with the console minimized
#>
param([switch]$Minimized)

$root = $PSScriptRoot
$target = Join-Path $root "start_jadu.bat"
$icon = Join-Path $root "static\jadu.ico"
if (-not (Test-Path $target)) { throw "start_jadu.bat not found next to this script." }

$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
foreach ($dir in $places) {
    $path = Join-Path $dir "Jadu.lnk"
    $lnk = $shell.CreateShortcut($path)
    $lnk.TargetPath = $target
    $lnk.WorkingDirectory = $root
    $lnk.Description = "Jadu - your local personal AI assistant"
    if (Test-Path $icon) { $lnk.IconLocation = $icon }
    $lnk.WindowStyle = $(if ($Minimized) { 7 } else { 1 })
    $lnk.Save()
    Write-Host "Created $path"
}
