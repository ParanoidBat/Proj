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

def plotTimAmp(audio, rate):
     plt.plot(npy.arange(audio.shape[0])/rate,audio) # time-amplitude plot
     plt.title("Audio wave")
     plt.xlabel("Time - s")
     plt.ylabel("Amplitude")
     plt.show()
    
def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")
    plt.pcolormesh(time, freq/1000, 10*npy.log10(sx), cmap="viridis") #log the frequency axis for better readability
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")
    plt.title("Spectrogram - audio")
    #plt.ylim(0,10)
    plt.show()
    
    return freq, time, sx

def energyFrames(time, spectogram):
    average =[]
#    for i in range(time.size):
    i = 0
    while i < time.size:
        if i+4 >= time.size: break
        #take mean of matrix along x-axis
        vector = npy.mean(spectogram[:170,i:i+4],1) #frequencies after index 170 are useless. As 8k Hz is the last formant of use
        
        #get a single value out of the vector, which represents a single value energy magnitude of the frame
        vector_avg = npy.mean(vector)
        average.append(vector_avg)
        
        i+=5
    
    return average

def segment(energy_frames):
    #silent <1000, segment == 25000
    segments = []
    segment_vals = []
    centre_vals = [] #centered b/w ith and kth index, when extending segment
    seg_start = []
    i = k = 0
    
    while i < len(energy_frames):
        if energy_frames[i] >=25000.0:
            k = i
            #the segment should start where the plot begins to rise again. So when segment is found, traverse back to find a value
            #greater than previous one, and update the segment to start/stop at that value
            while 1:
                if energy_frames[k] < energy_frames[k-1]:
                    break
                else:
                    k-=1
            
            #store the segments
            del segment_vals[k+1:]
            segments.append(segment_vals[:])
            
            # handle index out of bounds case
            if not segment_vals:
                pass
            else:
                seg_start.append(segment_vals[0])
            
            centre_vals = energy_frames[k:i]  #prepend to next segment
            
            print ("segmented at ", energy_frames[k])
            
            if max(segment_vals) > 1000.0:
                print("non-silence")
            else:
                print("silence")
            
            del segment_vals[:]
            
            for j in range(i, len(energy_frames)):
                if energy_frames[j] < 25000.0:
                    k = j
                    
                    while 1:
                        if k+1 == len(energy_frames): break #dont wanna go out of bounds
                    
                        if energy_frames[k] < energy_frames[k+1]:
                            break
                        else:
                            k+=1
                    
                    #while loop breaks where the transition is found, create a new list with both segments' values
                    u_seg_vals = centre_vals[:] + segment_vals[:] + energy_frames[j:k+1]
                    
                    segments.append(u_seg_vals[:])
                    
                    if not u_seg_vals:
                        pass
                    else:
                        seg_start.append(u_seg_vals[0])
                    
                    print ("segmented at ", energy_frames[k])
                    print ("non-silence") #this condition can only be met if iterator has passed a value >25000, when going back,
                    #it's obvious the previous region was non-silence
                    
                    del segment_vals[:]
                    del u_seg_vals[:]
                    del centre_vals[:]
                    
                    i = k
                    break
                
                segment_vals.append(energy_frames[j])
                
        else: segment_vals.append(energy_frames[i])
        i+=1
        
    return segments, seg_start

def createPattern(segments):
    pattern = ""
    
    for seg in segments:
        if max(seg) > 1000 and max(seg) < 250000: #if energy is b/w 1k to 300k it's a consonant, else if <300k, it's a wovel
            pattern+="c"
        elif max(seg) >=250000: pattern +="v"
        
    return pattern

def plotStuff(data, title, xlabel, ylabel, xlim1= 0, xlim2= 0, ylim1 = 0, ylim2= 0): # a helper function to plot any data
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if xlim1 != 0 and xlim2 != 0:
        plt.xlim(xlim1,xlim2)
        
    if ylim1 != 0 and ylim2 != 0:
        plt.ylim(ylim1, ylim2)
        
    plt.show()

# as the project was made originally
def makingOf():
    #read audio files
    rate_bit, data_bit = read("bit.wav")
    
    #store audio samples in numpy array
    bit_array = npy.array(data_bit)
    
    audio = npy.mean(bit_array,1) #convert to mono channel and trim audio
    
    # plot with time, instead of samples at x-axis
    #sd.play(audio)
    #length = audio.shape[0]/rate_bat #length of audio (samples/sample rate)
    plotTimAmp(audio, rate_bit)
    
    # plot spectogram
    freq, time, sx = plotSpec(audio, rate_bit)
    
    #compute high density energy frames, take 5 element window
    ef = energyFrames(time, sx)
    plotStuff(ef, "Energy contour", "index", "energy")
    
    #recognize and get silence and non-silence regions
    segments, seg_start = segment()
    
    #get the pattern string
    pattern = createPattern()
    
    #plot segments over contour
    contour = npy.array(ef)
    seg_indices = [ef.index(i) for i in seg_start]
    indices = npy.array(seg_indices)
    
    plt.plot(contour)
    plt.plot(indices, contour[indices],'x')
    plt.title("Energy contour (segments marked)")
    plt.xlabel("Index")
    plt.ylabel("Energy")
    plt.show()
    
def RecognizeVowels(audio_sample):
    sample_rate, wave_data = read(audio_sample)
    data_array = npy.array(wave_data)
    audio = npy.mean(data_array,1)
    
    freq, time, sx = plotSpec(audio, sample_rate)
    
    ef = energyFrames(time, sx)
    
    segments, seg_start = segment(ef)
    
    pattern = createPattern(segments)
    
    contour = npy.array(ef)
    seg_indices = [ef.index(i) for i in seg_start]
    indices = npy.array(seg_indices)
    
    plt.plot(contour)
    plt.plot(indices, contour[indices],'x')
    plt.title("Energy contour (segments marked)")
    plt.xlabel("Index")
    plt.ylabel("Energy")
#    plt.ylim(250000,)
    plt.show()
    
    return pattern