import VowelRecognizer as vr
#from VowelRecognizer import npy
#from scipy.io.wavfile import read

pre = vr.Preprocessing()
pre.recognizeVowels("audio.wav", visual=True)

#data = uti.organize()

#uti.plotStuff(data, "organized", "time", "energy")