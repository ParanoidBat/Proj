#import sounddevice as sd
#import numpy as npy
#import scipy.io.wavfile as wav
import VowelRecognizer as vr
from VowelRecognizer import sd

sample_rate = 48000
duration = 3 # seconds

print("Listening for 3 secs. . .\n")
speech = sd.rec(duration*sample_rate, samplerate=sample_rate, channels=1, dtype='float64')
sd.wait()

#sd.play(speech)

pre = vr.Preprocessing();

pre.recognizeVowels(speech, sample_rate, visual=True)
print(pre.seg_start)