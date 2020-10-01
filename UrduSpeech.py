import VowelRecognizer as vr
#from VowelRecognizer import npy
#from scipy.io.wavfile import read

pre = vr.Preprocessing()
#pre.recognizeVowels("Samples/Kahan ho.wav", visual=True)
#
#data = pre.getEnergy()
#troughs =pre.getTroughs()
#crests = pre.getCrests()

with open("test.txt", "r") as file:
    for line in file:
        tmp = line.rstrip("\n").split(",")
        tmp2 = ""
        for e in tmp[-1:]: tmp2 += e
        
        del tmp[len(tmp) - 1 :]
        tmp = list(map(float, tmp))
        
        pre.plotStuff(tmp, "organized", "time", "energy")