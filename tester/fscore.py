import sys
import os
import pdb,re,numpy

if len(sys.argv) != 3:
	print "Usage : python fscore.py <user-file> <gold-file>"
	sys.exit(1)
#Files don't exist
if not(os.path.isfile(sys.argv[1]) and  os.path.isfile(sys.argv[2])):
	print "Error: Atleast one of file does not exist"
	sys.exit(1)

#Read both user and gold files
in_file_user = open(sys.argv[1],"r")
in_file_gold = open(sys.argv[2],"r")

lines_user = in_file_user.readlines()
lines_gold = in_file_gold.readlines()

num_lines_user=len(lines_user)
num_lines_gold=len(lines_gold)
in_file_user.close()
in_file_gold.close()

print numpy.mean(numpy.array(lines_user)==numpy.array(lines_gold))


# Number of lines don't match in two files
if num_lines_gold != num_lines_user : 
	print "Error : number of lines are not equal in two files"
	sys.exit()

# list of classes possible
classes_list = ['0','2','4']

# confusion matrix, rows are actual classes, columns are predicted classes
table = [[0 for i in range(len(classes_list))] for j in range(len(classes_list))]

for i in range(num_lines_user):
	user_label = lines_user[i].strip()
	gold_label = lines_gold[i].strip()
	actual_index = classes_list.index(gold_label)
	predicted_index = classes_list.index(user_label)
	table[actual_index][predicted_index] += 1

print 'confusion matrix : ',table	
all_precisions = []
all_recalls = []
for i in range(len(classes_list)):
	column_sum = sum(row[i] for row in table)	
	if column_sum == 0:
		precision = 1.0
	else:
		precision = table[i][i]/float(column_sum)
	all_precisions.append(precision)
	row_sum = sum(table[i])
	if row_sum == 0:
		recall = 1.0
	else:
		recall = table[i][i]/float(row_sum)
	all_recalls.append(recall)

num_classes = len(classes_list)
macroPrecision=sum(all_precisions[:num_classes])/num_classes
macroRecall=sum(all_recalls[:num_classes])/num_classes

if((macroPrecision==0)and(macroRecall==0)):
	macroF_Score=0
else:
	macroF_Score=2*macroPrecision*macroRecall/(macroPrecision+macroRecall)

print "macroF_Score is %f, " %(macroF_Score) 
print "macroPrecision is %f" %(macroPrecision) 
print "macroRecall is %f" %(macroRecall)


