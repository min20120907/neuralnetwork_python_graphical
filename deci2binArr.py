import numpy as np
def deci2bin(num):
    binary=[]
    while num != 0:
        bit = num % 2
        binary.append(bit)
        num = num / 2
    return np.asarray(binary)