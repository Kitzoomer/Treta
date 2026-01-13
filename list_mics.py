import sounddevice as sd

for i, d in enumerate(sd.query_devices()):
    if d.get("max_input_channels", 0) > 0:
        print(i, "|", d["name"], "| inputs:", d["max_input_channels"], "| default_sr:", d.get("default_samplerate"))

