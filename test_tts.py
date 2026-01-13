import pyttsx3

engine = pyttsx3.init(driverName="sapi5")
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

voices = engine.getProperty("voices")
print("Voces:", len(voices))
for i, v in enumerate(voices):
    print(i, v.name)

engine.say("Hola Marian. Soy Treta. Si me oyes, el texto a voz funciona.")
engine.runAndWait()
print("TTS OK")
