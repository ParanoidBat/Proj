# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:50:07 2019

@author: Batman
"""
from scipy.io.wavfile import read
import numpy as npy
import matplotlib.pyplot as plt
#import sounddevice as sd
from scipy import signal
from collections import deque

def plotTimAmp(audio, rate):
    plt.plot(npy.arange(audio.shape[0])/rate_bit,audio) # time-amplitude plot
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Plot - bit')
    #plt.xlim(0.4,)
    #plt.ylim(900,)
    plt.show()
    
def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate_about, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
    plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis") #log the frequency axis for better readability
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram - audio")
    #plt.ylim(0,10)
    plt.show()
    
    return freq, time, sx

def plotEnergyContour(time, spectogram):
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
#    plt.xlim(200,250)
#    plt.ylim(0,30000) 
    plt.show()
    
    return average

def segment():
    #silent <1000, segment == 25000
    segments = []
    segment_vals = []
    con_list = []
    i = k = 0
    
    while i < len(average):
        if average[i] >=25000.0:
            k = i
            #the segment should start where the plot begins to rise again. So when segment is found, traverse back to find a value
            #greater than previous one, and update the segment to start/stop at that value
            while 1:
                if average[k] < average[k-1]:
                    break
                else:
                    k-=1
                    
            #while loop breaks where the transition is found, create a new list with both segments' values
            u_seg_vals = average[k:i] + segment_vals[:]
            
            #store the segments
            segments.append(u_seg_vals[:])
            print ("segmented at ", average[k]) #changes i to k
            
            if max(segment_vals) > 1000.0:
                print("non-silence")
            else:
                print("silence")
            
            del segment_vals[:]
            del u_seg_vals[:]
                
            for j in range(i, len(average)):
                if average[j] < 25000.0:
                    k = j
                    
                    while 1:
                        if k+1 == len(average): break #dont wanna go out of bounds
                    
                        if average[k] < average[k+1]:
                            break
                        else:
                            k+=1
                    
                    #while loop breaks where the transition is found, create a new list with both segments' values
                    u_seg_vals = segment_vals[:] + average[j+1:k+1] #include next from j, include k
                    
                    segments.append(u_seg_vals[:])
                    print ("segmented at ", average[k])
                    print ("non-silence") #this condition can only be met if iterator has passed a value >25000, when going back,
                    #it's obvious the previous region was non-silence
                    
                    del segment_vals[:]
                    del u_seg_vals[:]
                    i = j
                    break
                
                segment_vals.append(average[j])
                
        else: segment_vals.append(average[i])
        i+=1
        
    return segments

def createPattern():
    pattern = ""
    
    for seg in segments:
        if max(seg) > 1000 and max(seg) < 300000:
            pattern+="c"
        elif max(seg) >=300000: pattern +="v"
        
    return pattern

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
plotTimAmp(audio, rate_bit)

# plot spectogram
freq, time, sx = plotSpec(audio, rate_bit)

#plot high density energy frames, take 5 element window
average = plotEnergyContour(time, sx)

#recognize and get silence and non-silence regions
segments = segment()

#get the pattern string
pattern = createPattern()