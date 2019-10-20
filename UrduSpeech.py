# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 21:30:17 2019

@author: Batman
"""

import VowelRecognizer as vr

pattern = []

#pattern.append(vr.RecognizeVowels("kahan ho.wav"))
#pattern.append(vr.RecognizeVowels("bas aa raha hn.wav"))
#pattern.append(vr.RecognizeVowels("kya hal hai.wav"))
#pattern.append(vr.RecognizeVowels("mae theek hn.wav"))
#pattern.append(vr.RecognizeVowels("message kholo.wav"))
#pattern.append(vr.RecognizeVowels("phone karo.wav"))
#pattern.append(vr.RecognizeVowels("salam.wav"))
#pattern.append(vr.RecognizeVowels("wasalam.wav"))
#pattern.append(vr.RecognizeVowels("whatsapp chalao.wav"))

#pattern.append(vr.RecognizeVowels("kahan ho2.wav"))
#pattern.append(vr.RecognizeVowels("bas aa raha hn2.wav"))
#pattern.append(vr.RecognizeVowels("kya hal hai2.wav"))
#pattern.append(vr.RecognizeVowels("mae theek hn2.wav"))
#pattern.append(vr.RecognizeVowels("message kholo2.wav"))
#pattern.append(vr.RecognizeVowels("phone karo2.wav"))
#pattern.append(vr.RecognizeVowels("salam2.wav"))
#pattern.append(vr.RecognizeVowels("wasalam2.wav"))
#pattern.append(vr.RecognizeVowels("whatsapp chalao2.wav"))

pattern.append(vr.RecognizeVowels("kahan ho3.wav"))
pattern.append(vr.RecognizeVowels("bas aa raha hn3.wav"))
pattern.append(vr.RecognizeVowels("kya hal hai3.wav"))
pattern.append(vr.RecognizeVowels("mae theek hn3.wav"))
pattern.append(vr.RecognizeVowels("message kholo3.wav"))
pattern.append(vr.RecognizeVowels("phone karo3.wav"))
pattern.append(vr.RecognizeVowels("salam3.wav"))
pattern.append(vr.RecognizeVowels("wasalam3.wav"))
pattern.append(vr.RecognizeVowels("whatsapp chalao3.wav"))

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