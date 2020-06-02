import numpy as npy

indices = {"s": 0, "v": 1, "c": 2, "f": 3}

data = npy.zeros((340, 278)) # get data
prop = npy.zeros((340, 4)) # 340 sampling rows, for 4 properties each (s, c, v, f)

with open ("ef_peaks.txt", "r") as file:
    for line in file:
        tmp = line.rstrip("\n").split(",")
        tmp2 = str(tmp[-1:])
        
        del tmp[len(tmp) - 1 :]
        tmp = list(map(float, tmp))
        
        # set the input vector
        for i in range(340):
            for j in range(278):
                data[i,j] = tmp[j]
            
            # set output vector
            if tmp2.startswith( "s") :
                print("s hai\n")
                prop[i,0] = 1
            elif tmp2 in "v":
                print("v hai\n")
                prop[i,1] = 1
            elif tmp2 in "c":
                print("c hai\n")
                prop[i,2] = 1
            elif tmp2 in "f":
                print("f hai\n")
                prop[i,3] = 1