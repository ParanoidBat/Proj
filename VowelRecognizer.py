# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
<<<<<<< Updated upstream
from python_speech_features import base
=======
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
#mel-frequency cepstral co-efficients (MFCC)
freqs, px = signal.welch(audio, fs= rate_bit, window= "hamming", nperseg=400, noverlap=(400-160), nfft= 512, detrend= False, scaling= "spectrum")
#plt.semilogy(freqs, px) #to plot
#plt.xlabel('frequency [Hz]')
#plt.ylabel('PSD [V**2/Hz]')

#compute mel filterbank
#convert frequency to mel scale
#(http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn1)
lower_frequency = 1125*npy.log(1 + (300/700)) #300Hz is standard
higher_frequency = 1125*npy.log(1 + ((audio.size/2)/700)) #half the sampling frequency
filters = npy.linspace(lower_frequency, higher_frequency, 12) #taking 10 filters, need 12 points for that. standard filters are 26-40

#convert filters back to freq
#(http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/#eqn2)
filters_hz = []
for i in range(filters.size):
    filters_hz.append(700*(npy.exp(filters[i]/1125)-1))

#We don't have the frequency resolution required to put filters at the exact points calculated above,
#so we need to round those frequencies to the nearest FFT bin.
fft_bin = []
for i in range(len(filters_hz)):
    fft_bin.append(floor((512+1)*filters_hz[i]/rate_bit)) #floor((nfft+1)*h(i)/samplerate)

#create the filterbanks
fbank = npy.zeros([10,512//2+1])
for j in range(0,10):
    for i in range(int(fft_bin[j]), int(fft_bin[j+1])):
        fbank[j,i] = (i - fft_bin[j]) / (fft_bin[j+1]-fft_bin[j])
    for i in range(int(fft_bin[j+1]), int(fft_bin[j+2])):
        fbank[j,i] = (fft_bin[j+2]-i) / (fft_bin[j+2]-fft_bin[j+1])
        
plt.plot(fbank)
plt.show()
#
#freq, time, sx = signal.spectrogram(audio, fs=rate_bit, window="hamming", nperseg=1024, noverlap=(924), detrend=False, scaling="spectrum")
#pspec = freq**2 #powerspectrum
#energy= npy.sum(pspec)
#energy = npy.where(energy == 0,npy.finfo(float).eps,energy) # if energy is zero, we get problems with log
#
#features = npy.dot(pspec, fbank.T) #compute filterbank energies by taking ot product of power spectrum with transpose of filterbank
#features = npy.where(features == 0,npy.finfo(float).eps,features) # if feat is zero, we get problems with log
#features = npy.log(features)
#plt.plot(features)
#plt.show()

mfcc_features = base.mfcc(audio, rate_bit)
plt.plot(mfcc_features)
plt.show()

#delta_mfcc_features = base.delta(mfcc_features, 2)
#plt.plot(delta_mfcc_features)
#plt.show()
#
#fb_features = base.logfbank(audio, rate_bit)
#plt.plot(fb_features)
#plt.show()

#segmentation
#4096 frames (0.0853s) ka spectrogram bna k, uski freqs ka mean lena hai. puri wave ka
#segment = audio[:4096]
#freq, time, sx = signal.spectrogram(audio, fs=rate_about, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
#mean1 = npy.mean(freq)
#summation = []
#summation.append(0.0)
#j =0
#
#for i in range(audio.size):
#    i+=4095 #implicit loop step is 1, adding 4095 makes it 4096
#    segment = audio[i: i+4096]
#    if segment.size < 925: break
#
#    freq, time, sx = signal.spectrogram(segment, fs=rate_bit, window="hamming", nperseg=1024, noverlap=(924), detrend=False, scaling="spectrum")
#    mean2 = npy.mean(freq)
#    summation.append(summation[j] + abs(mean2 - mean1))
#    mean1 = mean2
#    j+=1
#
#summation = npy.array(summation)
#plt.plot(summation)
#plt.title("summation")
#plt.xlabel("elements")
#plt.ylabel("values")
##plt.xlim(24882, )
#plt.show()
=======
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
>>>>>>> Stashed changes
