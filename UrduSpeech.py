import VowelRecognizer as vr
from VowelRecognizer import npy
from scipy.io.wavfile import read

pre = vr.Preprocessing()

try:
    sample_rate, wave_data = read("audio.wav")
    raise ValueError
except Exception as e:
    print(e.__traceback__)

audio = npy.array(wave_data)

pre.recognizeVowels(audio, sample_rate, visual=True)