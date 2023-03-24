# Loop closure testing script
import numpy as np

np.seterr(all='raise')

with open('descriptors.txt', 'r') as file:
    lines = file.readlines()

descriptors = {}
for i, line in zip(range(len(lines)), lines):
    try:
        des = np.fromstring(line.strip('[]\n'), sep=';')
        descriptors[i] = des
    except:
        print(i)
        print(line)

print(len(descriptors))