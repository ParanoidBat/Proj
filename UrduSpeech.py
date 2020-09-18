import VowelRecognizer as vr
from VowelRecognizer import npy
import pyaudio

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000
RECORD_SECONDS = 2

p = pyaudio.PyAudio()

stream = p.open(format=1,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Listening for %d secs. ." %(RECORD_SECONDS))

speech = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    speech.append(npy.frombuffer(data))

numpy_speech = npy.hstack(speech)

stream.stop_stream()
stream.close()
p.terminate()

#speech = sd.rec(duration*sample_rate, samplerate=sample_rate, channels=2, dtype='float64')
#sd.wait()

#sd.play(speech)

pre = vr.Preprocessing()

pre.recognizeVowels(numpy_speech, RATE, True, True)