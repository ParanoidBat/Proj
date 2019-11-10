# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:30:17 2019

@author: Batman
"""

import VowelRecognizer as vr

pattern = []

#pattern.append(vr.recognizeVowels("kahan ho2.wav"))
#pattern.append(vr.recognizeVowels("bas aa raha hn2.wav"))
#pattern.append(vr.recognizeVowels("kya hal hai2.wav"))
#pattern.append(vr.recognizeVowels("mae theek hn2.wav"))
#pattern.append(vr.recognizeVowels("message kholo2.wav"))
#pattern.append(vr.recognizeVowels("phone karo2.wav"))
#pattern.append(vr.recognizeVowels("salam2.wav"))
#pattern.append(vr.recognizeVowels("wasalam2.wav"))
#pattern.append(vr.recognizeVowels("whatsapp chalao2.wav"))

pattern.append(vr.recognizeVowels("kahan ho3.wav"))
pattern.append(vr.recognizeVowels("bas aa raha hn3.wav"))
pattern.append(vr.recognizeVowels("kya hal hai3.wav"))
pattern.append(vr.recognizeVowels("mae theek hn3.wav"))
pattern.append(vr.recognizeVowels("message kholo3.wav"))
pattern.append(vr.recognizeVowels("phone karo3.wav"))
pattern.append(vr.recognizeVowels("salam3.wav"))
pattern.append(vr.recognizeVowels("wasalam3.wav"))
pattern.append(vr.recognizeVowels("whatsapp chalao3.wav"))

patterns = dict()
patterns["kahan ho"] = pattern[0]
patterns["bas aa raha hn"] = pattern[1]
patterns["kya hal hai"] = pattern[2]
patterns["mae theek hn"] = pattern[3]
patterns["message kholo"] = pattern[4]
patterns["phone karo"] = pattern[5]
patterns["salam"] = pattern[6]
patterns["wasalam"] = pattern[7]
patterns["whatsapp chalao"] = pattern[8]

for x,y in patterns.items():
    print(x,y)