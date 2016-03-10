import sys
import os
import pdb

if not(os.path.isfile(sys.argv[1]) and  os.path.isfile(sys.argv[2])):
	print "Error:n Atleast one of file does not exist"
	sys.exit(1)

in_file_test = open(sys.argv[1],"r")
in_file_output = open(sys.argv[2],"r")
lines_test = in_file_test.readlines()
lines_output = in_file_output.readlines()

if (len(lines_test)!=len(lines_output)):
	print "Error:Not Equal number of lines in two files"
	sys.exit(1)

for i in range(len(lines_output)):
	nOutput=int(lines_output[i])
	if((nOutput != 0) and (nOutput !=4)):
		print "Output Wrong iin line %d" %(i+1)
		sys.exit(1)

print " Correct Format"
		
		

