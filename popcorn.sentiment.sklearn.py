import re
import sys
import pandas as pd
import numpy as np

from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_path = 'labeledTrainData.tsv'
test_path = 'testData.tsv'
MIN = int(sys.argv[1])
RANGE = (1,int(sys.argv[2]))
if int(sys.argv[3]) == 0:
    FEATURES = None
else:
    FEATURES = int(sys.argv[3])

print 'Min df:', sys.argv[1], ' Max ngram:', sys.argv[2], 'Num features:', sys.argv[3]

def remove_nonletters(text):
    out = []
    for i in xrange(0, len(text)):
        out.append(re.sub('[^a-zA-Z]', ' ', text[i]))

    return out

def main():
    logit = LogisticRegression(penalty = 'l2', dual = True, tol = 0.0001, C = 1, fit_intercept = True, intercept_scaling = 1.0, class_weight = None, random_state = None)
    tcv = TfidfVectorizer(min_df = MIN,  max_features = FEATURES, strip_accents = 'unicode', analyzer = 'word', ngram_range = RANGE, lowercase = True, use_idf=1, smooth_idf = 1, sublinear_tf = 1)
    
    print 'loading and cleaning data ... '
    train = pd.read_csv(train_path, sep = '\t')['review'].values
    train = remove_nonletters(train)
    y = pd.read_csv(train_path, sep = '\t')['sentiment'].values
    
    test = pd.read_csv(test_path, sep = '\t')['review'].values
    test = remove_nonletters(test)
    PhraseId = pd.read_csv(test_path, sep = '\t')['id']
    
    phrase_all = np.concatenate((train,test))
    
    print 'fitting and transforming with vectorizer ... '
    tcv.fit(phrase_all)
    #tcv.fit(train)
    phrase_all = None
    
    x_train = tcv.transform(train)
    x_test = tcv.transform(test)
    train = None
    test = None
    print x_train.get_shape()
    
    print 'cross validation ... '
    print np.mean(cross_validation.cross_val_score(logit, x_train, y, cv = 5))
    
    print 'fitting logit ... '
    logit.fit(x_train, y)
    x_train = None
    
    print 'making predictions ... '
    preds = logit.predict(x_test)
    pred_df = pd.DataFrame(preds)
    pred_df = pd.concat((PhraseId, pred_df), axis = 1)
    pred_df.to_csv('logit.preds.csv', index = False, header = ['id', 'sentiment'])
    
if __name__ == '__main__':
    main()