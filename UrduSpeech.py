# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:30:17 2019

@author: Batman
"""

import VowelRecognizer as vr

pattern = []

pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao2.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao3.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao4.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao5.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao6.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao7.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao8.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao9.wav"))
pattern.append(vr.recognizeVowels("Samples/Whatsapp chalao10.wav"))

#patterns = dict()
#patterns["kahan ho"] = pattern[0]
#patterns["bas aa raha hn"] = pattern[1]
#patterns["kya hal hai"] = pattern[2]
#patterns["mae theek hn"] = pattern[3]
#patterns["message kholo"] = pattern[4]
#patterns["phone karo"] = pattern[5]
#patterns["salam"] = pattern[6]
#patterns["wasalam"] = pattern[7]
#patterns["whatsapp chalao"] = pattern[8]
#
#for x,y in patterns.items():
#    print(x,y)