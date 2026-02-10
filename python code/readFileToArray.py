import numpy as np

def readFile(fileName,usage):
    #Python program to read a text file into array
    file = np.loadtxt(fileName, delimiter=',' ,dtype=float)
    return file