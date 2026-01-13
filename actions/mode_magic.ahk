; === TRETA · MODO MAGIC ===

urls := []
urls.Push("https://www.youtube.com/watch?v=0p6UidTS7Ao")
urls.Push("https://lifecounter.app/?force=true")

; Abrir URLs en navegador por defecto
for index, url in urls {
    Run, %url%
    Sleep, 2000
}

; Esperar a que el navegador esté en foco
Sleep, 4000

; Pantalla completa
Send, {F11}
