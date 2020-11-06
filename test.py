def file_len(*args):
    total = 0
    
    for fn in args:
        with open(fn) as f:
            for i, l in enumerate(f):
                pass
        
        total += (i+1)
        
    return total

i = file_len("crestsrt.txt", "troughsrt.txt")