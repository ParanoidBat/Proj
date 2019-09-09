# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
from scipy.io.wavfile import read
import numpy as npy
import matplotlib.pyplot as plt
#import sounddevice as sd
#from scipy import fftpack
from scipy import signal
from math import floor

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
#sd.play(audio)
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

#plot high density energy frames, take 5 element window
average =[]
for i in range(time.size):
    if i+4 >= time.size: break
    #take mean of matrix along x-axis
    vector = npy.mean(sx[:170,i:i+4],1) #frequencies after index 170 are useless. As 8k Hz is the last formant of use
    i+=5
    #get a single value out of the vector, which represents a single value energy magnitude of the frame
    vector_avg = npy.mean(vector)
    average.append(vector_avg)
    

plt.plot(average)
plt.title("Energy contour")
plt.xlabel("index")
plt.ylabel("energy")
#plt.xlim(0,20 )
plt.ylim(0,30000)
plt.show()

#recognize silence and non-silence regions
#silent <1000, segment == 25000
segment_vals = []
i =0

while i < len(average):
    if average[i] >=25000.0:
        print ("segmented at ", average[i])
        if max(segment_vals) > 1000.0:
            print("non-silence")
            del segment_vals[:]
        else:
            print("silence")
            del segment_vals[:]
            
        for j in range(i, len(average)):
            if average[j] < 25000.0:
                print ("segmented at ", average[j])
                print ("non-silence")
                del segment_vals[:]
                i = j
                break
            
            segment_vals.append(average[j])
            
    else: segment_vals.append(average[i])
    i+=1
