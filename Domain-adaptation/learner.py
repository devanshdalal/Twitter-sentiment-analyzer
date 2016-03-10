import re,sys
import os,csv
import numpy as np
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
import pickle
import itertools
import string
from sklearn.linear_model import SGDClassifier
import subprocess
import random

#assigning predictor and target variables

tweets =[]
labels = []
datafile='tmp.txt'

os.system('python shuffle.py '+sys.argv[1]+ ' '+datafile+' '+ sys.argv[2] )

with open(datafile, 'rb') as f:
	data = f.readlines();
	for x in data:
		labels.append(int(x[1]))
		tweets.append( unicode(x[5:-2],errors='ignore') )
f.close();

os.system('python shuffle.py mytraining.csv'+ ' '+datafile+' 0.25')

with open(datafile, 'rb') as f:
	data = f.readlines();
	for x in data:
		labels.append(int(x[1]))
		tweets.append( unicode(x[5:-2],errors='ignore') )
f.close();

os.system('rm '+datafile)

bad_words=['nautanki','chuti']

def remove_duplicates(t):
	res=''
	for i,x in enumerate(t):
		if (i==0 or t[i-1]!=x):
			res+=x;
	return res

def convert(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1)

def remove(t):
	res=''
	for i,x in enumerate(t):
		if x not in '#@,;.:' :
			res+=x;
	return res

def processTweet(tweet):
	# print tweet
	#Convert www.* or https?://* to URL
	tweet=re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','URL',tweet)
	# print tweet
	tweet=convert(tweet)
	# tweet=' '.join( (word.lower()+' '+word.lower() if len(word)>=3 and word.isupper() else word) for word in tweet.split() )
	tweet=remove_duplicates(tweet)
	tweet=remove(tweet)
	tweet=tweet.strip()
	# print tweet
	return tweet

logfile = open('dump.txt','wb')

tweets = [processTweet(t) for t in tweets];

N =852
for i,x in enumerate(tweets):
	res=''
	if i<N:
		for j,y in enumerate(x):
			res+=' G'+y+' T'+y
	else:
		for j,y in enumerate(x):
			res+=' G'+y+' S'+y
	tweets[i]=res


test_labels,labels = labels[:N/10],labels[N/10:]
test_tweets,tweets = tweets[:N/10],tweets[N/10:]

######################################### LOGISTIC REGRESSION ########################################################

vectorizer = TfidfVectorizer( ngram_range=(1, 2), min_df=1) #,stop_words=stopwords)

for i,x in enumerate(test_tweets):
	res=''
	for j,y in enumerate(x):
		res+=' G'+y+' T'+y
	test_tweets[i]=res

train_vectors = vectorizer.fit_transform(tweets)
test_vectors = vectorizer.transform(test_tweets)

# logreg = SGDClassifier( loss='log', alpha=0.000001, penalty='l2' , n_iter=5, shuffle=True);
logreg = linear_model.LogisticRegression()
logreg.fit(train_vectors, labels)

f = open("GOLD.txt","wb")
f.write("\n".join([str(x) for x in test_labels]) )
f.close()

Z = logreg.predict(test_vectors)

# for i,x in enumerate(Z):
# 	if x==2 and random.random()<0.5:
# 		Z[i]=0;

f = open("MY.txt","wb")
f.write("\n".join([str(x) for x in Z]) )
f.close()

print subprocess.check_output( 'python fscore.py GOLD.txt MY.txt' ,shell=True)

logfile.close()

# Saving the objects:
with open('objs.pickle'+sys.argv[2], 'w') as f:
    pickle.dump([vectorizer,logreg], f)