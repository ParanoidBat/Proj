# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:30:17 2019

@author: Batman
"""

import VowelRecognizer as vr

pattern = []

pattern.append(vr.recognizeVowels("Samples/kahan ho.wav", 1.8))
pattern.append(vr.recognizeVowels("Samples/bas aa raha hn.wav", 1.8))
pattern.append(vr.recognizeVowels("Samples/kya hal hai.wav", 1.8))
pattern.append(vr.recognizeVowels("Samples/mae theek hn.wav", 1.9))
pattern.append(vr.recognizeVowels("Samples/message kholo.wav", 1.8))
pattern.append(vr.recognizeVowels("Samples/phone karo.wav", 1.9))
pattern.append(vr.recognizeVowels("Samples/salam.wav", 1.9))
pattern.append(vr.recognizeVowels("Samples/wasalam.wav", 1.8))
pattern.append(vr.recognizeVowels("Samples/whatsapp chalao.wav", 1.9))

#pattern.append(vr.recognizeVowels("Samples/kahan ho2.wav"))
#pattern.append(vr.recognizeVowels("Samples/bas aa raha hn2.wav"))
#pattern.append(vr.recognizeVowels("Samples/kya hal hai2.wav"))
#pattern.append(vr.recognizeVowels("Samples/mae theek hn2.wav"))
#pattern.append(vr.recognizeVowels("Samples/message kholo2.wav"))
#pattern.append(vr.recognizeVowels("Samples/phone karo2.wav"))
#pattern.append(vr.recognizeVowels("Samples/salam2.wav"))
#pattern.append(vr.recognizeVowels("Samples/wasalam2.wav"))
#pattern.append(vr.recognizeVowels("Samples/whatsapp chalao2.wav"))

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