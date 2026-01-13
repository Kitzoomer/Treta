Write-Host "=== TRETA · MODO TRABAJO ==="

$firefox = "$env:ProgramFiles\Mozilla Firefox\firefox.exe"
if (-not (Test-Path $firefox)) {
    $firefox = "${env:ProgramFiles(x86)}\Mozilla Firefox\firefox.exe"
}

$urls = @(
  "https://chat.openai.com",
  "https://mail.google.com",
  "https://calendar.google.com/calendar/u/0/r",
  "https://www.youtube.com/watch?v=jfKfPfyJRdk"
)

if (Test-Path $firefox) {
    Start-Process -FilePath $firefox -ArgumentList $urls
}

Write-Host "Modo Trabajo activado."
