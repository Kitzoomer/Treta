# =========================
# TRETA - Crear estructura Panel de Control
# SHUTDOWN DESACTIVADO
# =========================

$Base = "C:\treta"

# Carpetas
$folders = @(
  "$Base\actions",
  "$Base\bridge",
  "$Base\data"
)

foreach ($f in $folders) {
  New-Item -ItemType Directory -Force -Path $f | Out-Null
}

# Bridge
New-Item -ItemType File -Force -Path "$Base\bridge\log.txt" | Out-Null
New-Item -ItemType File -Force -Path "$Base\bridge\pending_confirm.json" | Out-Null

# Estado inicial
@"
{
  "mood": "calm",
  "last_interaction": "2025-01-01T00:00:00",
  "daily_interactions": 0,
  "mode": "normal"
}
"@ | Set-Content -Encoding UTF8 "$Base\data\state.json"

New-Item -ItemType File -Force -Path "$Base\data\ideas.jsonl" | Out-Null
New-Item -ItemType File -Force -Path "$Base\data\diary.jsonl" | Out-Null

# Actions (placeholders seguros)
@"
Write-Host 'Treta presence start (placeholder)'
"@ | Set-Content -Encoding UTF8 "$Base\actions\presence_start.ps1"

@"
Write-Host 'Treta sleep (placeholder)'
"@ | Set-Content -Encoding UTF8 "$Base\actions\presence_sleep.ps1"

@"
Write-Host 'Modo dibujo (placeholder)'
"@ | Set-Content -Encoding UTF8 "$Base\actions\mode_draw.ps1"

@"
Write-Host 'Modo caos (placeholder)'
"@ | Set-Content -Encoding UTF8 "$Base\actions\mode_chaos.ps1"

@"
Write-Host 'SHUTDOWN DESACTIVADO (placeholder seguro)'
Write-Host 'Para activarlo algún día: shutdown /s /t 0'
"@ | Set-Content -Encoding UTF8 "$Base\actions\shutdown_safe.ps1"

Write-Host ""
Write-Host "✅ Treta Panel creado correctamente en C:\treta" -ForegroundColor Green
Write-Host "🛡️ Shutdown DESACTIVADO (seguro)" -ForegroundColor Yellow
