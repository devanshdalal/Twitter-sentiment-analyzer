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
weights=[]
datafile='tmp.txt'
N =852

os.system('python shuffle.py '+sys.argv[1]+ ' '+datafile +' 1' )

with open(datafile, 'rb') as f:
	data = f.readlines();
	for x in data:
		# labels.append(int(x[1]))
		tweets.append( unicode(x[0:-1],errors='ignore') )
		weights.append(1.0)
f.close();

# os.system('python shuffle.py mytraining.csv'+ ' '+datafile+' '+sys.argv[2])

# with open(datafile, 'rb') as f:
# 	data = f.readlines();
# 	for x in data:
# 		labels.append(int(x[1]))
# 		tweets.append( unicode(x[5:-2],errors='ignore') )
# 		weights.append(2.0/500.0)
# f.close();

os.system('rm '+datafile)

# bad_words=['nautanki','chuti']

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
		if x not in ',;.:`$%&*\"\'' :
			res+=x;
	return res

def processTweet(tweet):
	# print tweet
	#Convert www.* or https?://* to URL
	# tweet=re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','URL',tweet)
	# print tweet
	tweet=convert(tweet)
	# tweet=' '.join( (word.lower()+' '+word.lower() if len(word)>=3 and word.isupper() else word) for word in tweet.split() )
	tweet=remove_duplicates(tweet)
	tweet=remove(tweet)
	tweet=tweet.strip()
	# print tweet
	return tweet

tweets = [processTweet(t) for t in tweets];

for i,x in enumerate(tweets):
	res=''
	for j,y in enumerate(x):
		res+=' G'+y+' T'+y
	tweets[i]=res


# test_labels,labels = labels[:N/10],labels[N/10:]
# test_tweets,tweets = tweets[:N/10],tweets[N/10:]

######################################### LOGISTIC REGRESSION ########################################################

# vectorizer = HashingVectorizer( ngram_range=(1, 2), binary=True) #,stop_words=stopwords)
# vectorizer = TfidfVectorizer( ngram_range=(1, 2), min_df=1,binary=True) #,stop_words=stopwords)
# vectorizer = CountVectorizer( ngram_range=(1, 2), min_df=1,binary=True) #,stop_words=stopwords)

# for i,x in enumerate(test_tweets):
# 	res=''
# 	for j,y in enumerate(x):
# 		res+=' G'+y+' T'+y
# 	test_tweets[i]=res

with open('objs.pickle0.5','rb') as f:
    vectorizer,logreg = pickle.load(f)

tweets=np.array(tweets)

test_vectors = vectorizer.transform(tweets)

Z = logreg.predict(test_vectors)

f = open(sys.argv[2],"wb")
f.write("\n".join([str(x) for x in Z]) )
f.close()