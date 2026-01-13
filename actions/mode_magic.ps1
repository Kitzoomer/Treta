Write-Host "=== TRETA · MODO MAGIC (HUB) ==="

cd /d C:\treta

# Activar venv si existe
if (Test-Path ".\.venv\Scripts\activate") {
  call .\.venv\Scripts\activate
}

python .\magic_hub.py
