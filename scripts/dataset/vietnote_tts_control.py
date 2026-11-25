"""eSpeak NG rule-based control for the existing Python TTS backend."""

import ctypes as C
import wave
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def engine():
    import espeakng_loader

    lib = C.CDLL(espeakng_loader.get_library_path())
    lib.espeak_Initialize.argtypes = [C.c_int, C.c_int, C.c_char_p, C.c_int]
    lib.espeak_SetVoiceByName.argtypes = [C.c_char_p]
    lib.espeak_SetParameter.argtypes = [C.c_int, C.c_int, C.c_int]
    lib.espeak_Synth.argtypes = [C.c_void_p, C.c_size_t, C.c_uint, C.c_int, C.c_uint, C.c_uint, C.c_void_p, C.c_void_p]
    lib.espeak_SetSynthCallback.argtypes = [C.CFUNCTYPE(C.c_int, C.POINTER(C.c_short), C.c_int, C.c_void_p)]
    data = Path(espeakng_loader.get_data_path()).parent
    # Synchronous retrieval returns PCM through the callback and never uses the sound device.
    rate = lib.espeak_Initialize(2, 0, str(data).encode("utf-8"), 0x8000)
    if rate <= 0:
        raise RuntimeError("eSpeak NG could not load its phoneme data")
    return lib, rate


def synth(text, voice, out_path):
    lib, rate = engine()
    if lib.espeak_SetVoiceByName(str(voice).encode()) or lib.espeak_SetParameter(1, 160, 0):
        raise ValueError(f"Unsupported eSpeak voice: {voice}")
    pcm = bytearray()
    callback_type = lib.espeak_SetSynthCallback.argtypes[0]

    @callback_type
    def collect(samples, count, _events):
        if samples and count:
            pcm.extend(C.string_at(samples, count * C.sizeof(C.c_short)))
        return 0

    lib.espeak_SetSynthCallback(collect)
    data = C.create_string_buffer(text.encode("utf-8"))
    if lib.espeak_Synth(data, C.sizeof(data), 0, 1, 0, 1, None, None) != 0 or not pcm:
        raise RuntimeError("eSpeak produced no speech")
    with wave.open(str(out_path), "wb") as out:
        out.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        out.writeframes(pcm)
