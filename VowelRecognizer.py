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

audio = npy.mean(bit_array[20000:50000:],1) #convert to mono channel and trim audio

# plot with time, instead of samples at x-axis
sd.play(bit_array,60000)
#length = audio.shape[0]/rate_bat #length of audio (samples/sample rate)
plt.plot(npy.arange(audio.shape[0])/rate_bat,audio) # time-amplitude plot
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Plot - Bat')
#plt.xlim(0.4,)
#plt.ylim(900,)
plt.show()
#this is a useless lone
# plot spectogram
freq, time, sx = signal.spectrogram(audio, fs=rate_about, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (kHz)")
plt.title("Spectrogram - audio")
#plt.ylim(0,10)
plt.show()
