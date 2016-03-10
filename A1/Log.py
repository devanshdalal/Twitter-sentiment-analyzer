import re,sys
import pandas
import numpy as np
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm,linear_model,ensemble
import pickle

#assigning predictor and target variables

colnames = ['label', 'tweet']
data = pandas.read_csv(sys.argv[1], names=colnames)
stopwords = ['0', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves'];

labels = data.label.tolist()
tweets = data.tweet.tolist()

#start process_tweet
logfile = open('dump.txt','wb')

NEGATE = ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
              "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
              "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
              "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
              "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
              "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
              "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
              "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def processTweet(tweet):
    # process the tweets
    #Replace WORD with word word
    tweet=' '.join( (word.lower()+' '+word.lower() if len(word)>=3 and word.isupper() else word) for word in tweet.split() )
    #Replace negative words with NOT
    tweet=' '.join( ('not' if word in NEGATE else word) for word in tweet.split() )
    #trim punctuations
    tweet = re.sub(r'[\'",.;?]',' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #Remove duplicate chars
    tweet = re.sub(r'(.)\1+', r'\1\1',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','ATUSER',tweet)
    #Convert www.* or https?://* to URL
    tweet = re.sub(r'((https?://[^\s]+)|(www\.[^\s]+))','LINK',tweet)
    print >> logfile,  tweet
    return tweet
#end

tweets = [processTweet(t) for t in tweets];

test_labels,labels = labels[int(0.9*len(labels)):],labels[:int(0.9*len(labels))]
test_tweets,tweets = tweets[int(0.9*len(tweets)):],tweets[:int(0.9*len(tweets))]

#########################################################################################################
# LOGISTIC REGRESSION

vectorizer = TfidfVectorizer(ngram_range=(1,2),use_idf=True,m) #,stop_words=stopwords)

train_vectors = vectorizer.fit_transform(tweets)
test_vectors = vectorizer.transform(test_tweets)

logreg = linear_model.LogisticRegression(random_state=42)
logreg.fit(train_vectors, labels)

Z = logreg.predict(test_vectors)

s = np.array(test_labels) == np.array(Z)
print 'AccuracyLogRegression', logreg.score(test_vectors,test_labels);

# Saving the objects:
with open('objs.pickle', 'w') as f:
    pickle.dump([vectorizer,logreg], f)

exit(0)