import VowelRecognizer as vr
from VowelRecognizer import npy
import scipy
from VowelRecognizer import read


#with open("audio.wav", 'rb') as file:
#    print(file.read())

#pre = vr.Preprocessing()
#
#
#sample_rate, wave_data = read("audio.wav")
#audio = npy.array(wave_data)
#pre.recognizeVowels(audio, sample_rate)

print(scipy.__version__)