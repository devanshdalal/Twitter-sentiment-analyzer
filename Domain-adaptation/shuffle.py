import random,sys

with open(sys.argv[1], 'rb') as infile:
    lines = infile.readlines()

random.shuffle(lines)

with open(sys.argv[2], 'wb') as outfile:
    outfile.writelines( ''.join(lines[:int(float(sys.argv[3])*len(lines))] ) )