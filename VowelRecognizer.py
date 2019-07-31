# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
#import librosa
from scipy.io.wavfile import read
import numpy as npy
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import fftpack
from scipy import signal

#read audio files
rate_names, data_names = read("names.wav")
rate_bit, data_bit = read("bit.wav")
rate_bat, data_bat = read("bat.wav")
rate_about, data_about = read("about.wav")

#store audio samples in numpy array
names_array = npy.array(data_names)
bit_array = npy.array(data_bit)
bat_array = npy.array(data_bat)
about_array = npy.array(data_about)

audio = npy.mean(bit_array[20000:50000],1) #convert to mono channel and trim audio

# plot with time, instead of samples at x-axis
#sd.play(bit_array)
#length = audio.shape[0]/rate_bat #length of audio (samples/sample rate)
plt.plot(npy.arange(audio.shape[0])/rate_bat,audio) # time-amplitude plot
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Plot - bit')
#plt.xlim(0.4,)
#plt.ylim(900,)
plt.show()

# plot spectogram
freq, time, sx = signal.spectrogram(audio, fs=rate_about, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (kHz)")
plt.title("Spectrogram - audio")
#plt.ylim(0,10)
plt.show()

# 8 elements - 0.0853s k hisab se jitne elements bante haen time mae, utne freq k elements ka mean-
local_sum = prev_ls =0.0
counter = 0
j = 0

#while local_sum - prev_ls < 0.0853:
#        local_sum += time[counter]
#        counter+=1

#prev_ls = local_sum
#mean1 = npy.mean(freq[0:counter-1])
absolute = []
summation = []
summation.append(0.0)

for i in range(time.size):
    i+=counter
    
    while local_sum - prev_ls < 0.0853:
        if counter == time.size: break
        local_sum += time[counter]
        counter+=1
    
    if i >=time.size or i+(counter-2) >= time.size: break
#    prev_ls = local_sum
    rise = (npy.sum(freq[i+(counter-2)]-freq[i])) #the wanted index is -2 from the counter, cuz it's been +1 AND it's counting elements. not indices
#    absolute.append(abs(mean2 - mean1))
    run = (npy.sum(time[i+(counter-2)]-time[i]))
    summation.append(summation[j] + rise/run) #gradient of spectrogram
#    mean1 = mean2
    j+=1
    
summation = npy.array(summation)
plt.plot(sx)
plt.title("summation")
plt.xlabel("elements")
plt.ylabel("values")
plt.xlim(0,15 )
plt.show()