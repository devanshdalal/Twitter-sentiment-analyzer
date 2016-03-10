import re,sys
import pandas
#Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer,TfidfVectorizer
from sklearn import svm
from sklearn import linear_model

# X= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
# Y = np.array([3, 3, 3, 3, 4, 4, 3, 4, 3, 4, 4, 4])

#assigning predictor and target variables

colnames = ['label', 'tweet']
data = pandas.read_csv(sys.argv[1], names=colnames)

labels = data.label.tolist()
tweets = data.tweet.tolist()

#start process_tweet
def processTweet(tweet):
    # process the tweets
    #Convert to lower case
    tweet = tweet.lower()
    #Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    #Remove additional white spaces
    tweet = re.sub('[\s]+', ' ', tweet)
    #Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    #trim
    tweet = tweet.strip('\'"')
    return tweet
#end

# tweets = [processTweet(t) for t in tweets];

test_labels,labels = labels[int(0.9*len(labels)):],labels[:int(0.9*len(labels))]
test_tweets,tweets = tweets[int(0.9*len(tweets)):],tweets[:int(0.9*len(tweets))]

# vectorizer = CountVectorizer(min_df=10)
use_hashing=False;
if use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,n_features=200)
else:
    vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(tweets)
# classifier_rbf = svm.SVC()
# classifier_rbf.fit(train_vectors, train_labels)
# prediction_rbf = classifier_rbf.predict(test_vectors)

#Create a Gaussian Classifier
model = BernoulliNB()

# Train the model using the training sets 
model.fit(X, labels)


#Predict Output 
test_X = vectorizer.transform(test_tweets)

predicted = model.predict(test_X)
# print predicted
s = np.array(test_labels) == np.array(predicted)
print 'Accuracy', sum(s)*1.0/len(test_labels);

# ########################################################################################################

vectorizer = TfidfVectorizer()

train_vectors = vectorizer.fit_transform(tweets)
test_vectors = vectorizer.transform(test_tweets)

logreg = linear_model.LogisticRegression(C=1)
logreg.fit(train_vectors, labels)

Z = logreg.predict(test_vectors)

s = np.array(test_labels) == np.array(Z)
print 'AccuracyLogRegression', sum(s)*1.0/len(test_labels);

exit(0);
# ########################################################################################################

vectorizer = TfidfVectorizer(use_idf=True,
                             stop_words='english')

train_vectors = vectorizer.fit_transform(tweets)
test_vectors = vectorizer.transform(test_tweets)

classifier_rbf = svm.LinearSVC() # svm.LinearSVC()
classifier_rbf.fit(train_vectors, labels)
prediction_rbf = classifier_rbf.predict(test_vectors)

s = np.array(test_labels) == np.array(prediction_rbf)
print 'AccuracySVM', sum(s)*1.0/len(test_labels);
