from scipy.io.wavfile import read
from scipy.fftpack import fft
import numpy as npy
import sys
import matplotlib.pyplot as plt
import sounddevice as sd

rate_name, data_name = read("names.wav") #read audio
rate_about, data_about = read("about.wav")
rate_bat, data_bat = read("bat.wav")
rate_bit, data_bit = read("bit.wav")

print ("rate: " , rate_name)

npy.set_printoptions(threshold=sys.maxsize)

#print (numpy.array(a[1])) #print data. doesn't make difference even if 1 not mentioned

#names_array = npy.array(data_name) #store audio samples into numpy array
#
#data_bit = npy.mean(npy.array(data_bat))
#
#plt.plot(data_bit)
#plt.show()
#sd.play(data_bit)

#find peaks
#peaks, _= signal.find_peaks(audio, prominence=(1000,None), distance=4800)
#plt.plot(audio)
#plt.plot(peaks, audio[peaks], "x")
##plt.ylim(900,)
#print("peaks",peaks)


#get onsets
#a,ar = librosa.load("about.wav")
#onset_frames = librosa.onset.onset_detect(a,ar)
#plt.plot(a)
#plt.plot(onset_frames, a[onset_frames], "x")
##plt.ylim(900,)
#print("onset_frames",onset_frames)


#cepstrum
#powerspectrum = npy.abs(fftpack.fft(audio))**2
#cepstrum = fftpack.ifft(npy.log(powerspectrum))
#plt.plot(cepstrum)
