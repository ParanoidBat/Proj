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
#import glob
#import os
    
def plotSpec(audio, rate):
    freq, time, sx = signal.spectrogram(audio, fs=rate, window="hamming", nperseg=1024, noverlap=924, detrend=False, scaling="spectrum")    ;    return freq, time, sx

def energyFrames(time, spectogram):
    average =[]
    
    i = 0
    while i < time.size:
        if i+5 >= time.size: break
        #take mean of matrix along x-axis
        vector = npy.mean(spectogram[:170,i:i+5],1) #frequencies after index 170 are useless. As 8k Hz is the last formant of use
        
        #get a single value out of the vector, which represents a single value energy magnitude of the frame
        vector_avg = npy.mean(vector)
        average.append(vector_avg)
        
        i+=5
    
    return average

#segmentation of the data w.r.t peaks
def segment(energy_frames):
    # first segment == 25000. afterwards, find segments relatively; according to found local maxima
    segments = []
    segment_vals = []
    centre_vals = [] # centered b/w ith and kth index, when extending segment
    seg_indices = []
    peaks = []
    i = k = peak = 0 # main iterator, local iterator, local peak storage
    
    while i < len(energy_frames):
        if energy_frames[i] >=25000.0:
            k = i
            # the segment should start where the plot begins to rise. So when segment is found, traverse back to find a value
            #greater than previous one, and update the segment to start/stop at that value
            while 1:
                if energy_frames[k] < energy_frames[k-1]:
                    break
                else:
                    k-=1
            
            #store the segments
            del segment_vals[k+1:]
            segments.append(segment_vals[:])
            
            if segment_vals: # if list is not null
                seg_indices.append(energy_frames.index(segment_vals[0])) # get the index of the segment
            
            centre_vals = energy_frames[k:i]  # prepend to next segment
            
            del segment_vals[:]
            
        else:
            segment_vals.append(energy_frames[i])
            i+=1
            continue
            
        # next segmentations are done through different algorithm
        k = i
        
        while k < len(energy_frames):
            peak = energy_frames[k]
            while 1: # find local maxima/peak, break when found
                try:
                    k+=1
                    
                    if energy_frames[k] <= peak*0.6: # if next value is less than 2/3rd of value in 'peak', we've found the peak
                        peaks.append(energy_frames.index(peak))
                        break
                    
                    elif energy_frames[k] == peak or energy_frames[k] > peak: # if next value is greater than value in 'peak', update it (updation is also being done right below' while')
                        peak = energy_frames[k]
                
                except IndexError:
                    k-=1
                    break
                    
            # extend the segment to find a transition point
            while k < len(energy_frames):
                try:
                    if energy_frames[k] <= peak*0.6: # if the value is atleast 2/3rd of the peak
                        
                        # find transition point
                        while k < len(energy_frames):
                            if energy_frames[k] < energy_frames[k+1]: # if transition point is found
                                k+=1
                                raise IndexError # this is used as an escape from main while loop
                            else:
                                k+=1
                                
                    else:
                        k+=1
                    
                except IndexError:
                    k-=1 # get last index
                    break
                
            temp_seg = centre_vals[:] + energy_frames[i:k+1] if centre_vals else energy_frames[i:k+1] # k+1 is value right before plot begins to rise
            
            if centre_vals: del centre_vals[:]
            i = k = k+1
            
            if temp_seg:
                segments.append(temp_seg[:])
                seg_indices.append(energy_frames.index(temp_seg[0]))
                
                del temp_seg[:]
        
    return segments, seg_indices, peaks

#scale down the energies to give consistant data with the referenced data (eliminating effect of loud sound)
def scaleDown(scale_to, segment, peak):
    factor = scale_to/peak #get the factor to be multiplpied with the segment, to scale them down
    
    for i in range(len(segment)):
        segment[i] = segment[i]*factor

#smooth energy frames data to eliminate noise
def smooth(data, size, mean): 
    new_data = []
    factor = mean//2
    i = factor

    while(i< size-factor):
        new_data.append(sum(data[i-factor:i+factor])/mean) #take mean of 'mean' values, taking i'th value as the pivot
        i+=1
        
    return new_data


def zeroCrossingRate(wave):
    rate = []
    
    # taking 19ms window. 924 samples
    samples = 924
    
    above = True # boolean to check if value is above or below zero
    prev_state = above # remember the previous state to determine zero crossing
    
    try:
        while 1:
            # don't let go out of bounds
            if samples >= len(wave) or samples > len(wave) - 924:
                break
            
            counter = 0 # count number of times zero is crossed
            
            # set bool to determine crossing
            if wave[samples - 924] > 0:
                above = True
            else:
                above = False
                
            prev_state = above
            
            for i in range(samples - 924, samples):
                if wave[i] > 0:
                    above = True
                else:
                    above = False
                    
                if prev_state is not above: # the state changed, meaning zero is crossed
                    counter+=1
                    
                prev_state = above
                
            samples +=924
            rate.append(counter/0.019)
                
    except IndexError:
        rate.append(counter/0.019)
        print("Error: Index Is Out Of Bounds!")
        
    return npy.array(rate)


def zcrSmooth(zcr):
    # staright lines are plotted from peak to peak.
    
    indices = signal.find_peaks(zcr)[0]

    new_data = []
    
    for x in indices:
        new_data.append(zcr[x])
    
    return npy.array(new_data)

# get the factor to enable mapping of segments/peaks of energy frames to zcr
def calcMappingFactor(ef_length, zcr_length):
    return ef_length/zcr_length

# map segments/peaks from energy frames to zcr
def mapEnergyToZcr(factor, peaks, segments):
    indices = []
    
    # map peaks
    try:
        if peaks is None: raise IndexError
        
        for i in range(len(peaks)):
            indices.append(peaks[i]//factor)
    except IndexError:
        pass
    
    # map segments
    try:
        if segments is None: raise IndexError
        
        for i in range(len(segments)):
            indices.append(segments[i]//factor)
    except IndexError:
        pass
    
    return npy.array(indices, dtype = npy.int32)

# pad the data with 0's for scaling purpose to be used in neutral network
def padding(pad_with):
    new_list = [0]*pad_with
    return new_list

def writeToFile(_from_en, _to_en, energy, _from_zcr, _to_zcr, zcr, prop, filename):
    # max ef is 300
    pad = length = 0
    data = []
    
    with open(filename, "a") as file:        
        for f in range(len(_from_en)):
            # get length of segments fo energy+zcr            
            length = len(energy[_from_en[f] : _to_en[f]]) + len(zcr[_from_zcr[f] : _to_zcr[f]])
            
            if(length < 300): pad = 300 - length # if padding is needed

            data = padding(pad) # add padding
            
            for x in energy[_from_en[f] : _to_en[f]]: #appendd data from segment to list after padding
                data.append(x)
            for x in zcr[_from_zcr[f] : _to_zcr[f]]:
                data.append(x)
            
            for d in data:
                file.write(str(d) + ',') # write to file
            
            # reset varaibles
            del data[:] 
            length = pad = 0
#            file.write("\n") # comment out if labelling data
            
            try: file.write(prop[f] + "\n") # append property
            except: pass
    
    print("wrote to file data")

def writeSilences(from_en, to_en, energy, from_zcr, to_zcr, zcr, g):
    pad = length = 0
    data = []
    
    with open("silences.txt", "a") as file:
        for f in range(len(from_en)):

            length = len(energy[from_en[f] : to_en[f]]) + len(zcr[from_zcr[f] : to_zcr[f]])
            
            if(length < 300): pad = 300 - length # if padding is needed

            data = padding(pad) # add padding
            
            for x in energy[from_en[f] : to_en[f]]: #appendd data from segment to list after padding
                data.append(x)
            for x in zcr[from_zcr[f] : to_zcr[f]]:
                data.append(x)
            
            for d in data:
                file.write(str(d) + ',') # write to file
            
            # reset varaibles
            del data[:] 
            length = pad = 0
            
            # append property
            
            # going to silence. made this for cases in crest-to-crest labelling. when last vowel ends and proceeds to silence
            # only first case can be as such
            if g != "" and f == 0: file.write("g\n")
            
            else: file.write("s\n")
    
    print("wrote to file silence")

#####################
audio_sample = "Samples/Kahan ho15.wav"

sample_rate, wave_data = read(audio_sample)
data_array = npy.array(wave_data)
audio = npy.mean(data_array,1) #make it into mono channel

freq, time, sx = plotSpec(audio, sample_rate) #get spectrogram

ef = energyFrames(time, sx)

peak = max(ef) #if data's highest value is above threshold, scale it down
scale_to = 200000
if peak >scale_to:
    scaleDown(scale_to, ef, peak)

s_ef = smooth(ef, len(ef), 7)

segments, seg_start, peaks = segment(s_ef) #get segments, their indices and peaks' indices

zero_crossing_rate = zeroCrossingRate(audio)
smooth_zcr = npy.array(smooth(zero_crossing_rate, zero_crossing_rate.size, 7))

# get mapped values for zcr. pass one vector at a time
mapped_values = mapEnergyToZcr(calcMappingFactor(len(s_ef), smooth_zcr.size), 
                               None, seg_start) # peaks | seg_start

#plotting##
contour = npy.array(s_ef)
indices = npy.array(seg_start)
peakses = npy.array(peaks)

#mark segments and peaks on energy contour
plt.plot(contour)
plt.plot(indices, contour[indices],'x')
plt.plot(peakses, contour[peakses], 'v')
plt.title("Energy contour - "+ audio_sample)
plt.xlabel("Time - ds (deca sec)")
plt.ylabel("Energy")
#plt.ylim(0,1200)
#plt.xlim(2000,2500)
plt.show()

# zcr
plt.subplot(111)
plt.plot(smooth_zcr) # prev => zero_crossing_rate
plt.plot(mapped_values, smooth_zcr[mapped_values], 'x')
plt.title("Zero Crossing Rate - "+ audio_sample)
plt.xlabel("Time - hs (hecto sec)")
plt.ylabel("Rate")

plt.show()

from_en = [38, 57, 87]
to_en = [57, 87, 122]

from_zcr = [19, 29, 45]
to_zcr = [29, 45, 63]

prop = ["v", "v", "v"]

#writeSilences([144], [181], s_ef, [76], [95], smooth_zcr, "g")
writeToFile(from_en, to_en, s_ef, from_zcr, to_zcr, smooth_zcr, prop, "troughs.txt")