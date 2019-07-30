from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import numpy as npy
import sys
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.signal as sig

rate_name, data_name = read("names.wav") #read audio
rate_about, data_about = read("about.wav")
rate_bat, data_bat = read("bat.wav")
rate_bit, data_bit = read("bit.wav")

print ("rate: " , rate_name)

#npy.set_printoptions(threshold=sys.maxsize)

#print (numpy.array(a[1])) #print data. doesn't make difference even if 1 not mentioned

#names_array = npy.array(data_name) #store audio samples into numpy array
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
#norm = npy.hamming(audio.size)/sum(npy.hamming(audio.size)) #need to normalize hamming window before applying to signal
#filtered = signal.convolve(audio, norm) #to apply to wave, convolve it
#powerspectrum = npy.abs(fftpack.fft(filtered))**2 #get the power spectrum through fft over the windowed(filtered) signal
#cepstrum = fftpack.ifft(npy.log(powerspectrum))
#plt.plot(cepstrum)
#plt.show()

#mean 8 elements
#j = 0
#mean1 = npy.mean(freq[0:8])
#absolute = []
#summation = []
#summation.append(0.0)
#
#for i in range(freq.size):
#    i+=8
#    mean2 = (npy.mean(freq[i:i+7]))
#    absolute.append(abs(mean2 - mean1))    
#    summation.append(summation[j] + absolute[j])
#    mean1 = mean2
#    j+=1