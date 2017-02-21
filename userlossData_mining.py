from __future__ import division
import pandas as pd
import numpy as np

churn_df = pd.read_csv('D:\\working\\ML\\data\\churn.csv')
col_names = churn_df.columns.tolist()

print "Column names:"
print col_names

to_show = col_names[:6] + col_names[-6:]

print "\nSample data:"
churn_df[to_show].head(6)


churn_result = churn_df['Churn?']
y = np.where(churn_result == 'True.',1,0)

#we do not need these columns
to_drop = ['State', 'Area Code', 'Phone', 'Churn?']
churn_feat_space = churn_df.drop(to_drop, axis=1)

yes_no_cols = ["Int'l Plan","VMail Plan"]
churn_feat_space[yes_no_cols] = churn_feat_space[yes_no_cols] == 'yes'

features = churn_feat_space.columns

X = churn_feat_space.as_matrix().astype(np.float)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

print "Feature space hold %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)

print X[0]
print len(y[y == 0])

from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs) :
    #Construct a kfolds object
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_pred = y.copy()
     #Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_tain = y[train_index]
        #Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train, y_tain)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

def accuracy(y_true, y_pred):
    #numpy interprets True and False as 1 and 0
    return np.mean(y_true == y_pred)

print "Support vector machines:"
print "%.3f" % accuracy(y, run_cv(X,y,SVC))
print "Random forest:"
print "%.3f" % accuracy(y, run_cv(X,y,RF))
print "K-nearest-neighbors;"
print "%.3f" % accuracy(y, run_cv(X,y,KNN))

def run_prob_cv(X,y,clf_class,**kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]

        clf = clf_class(**kwargs)
        clf.fit(X_train, y_train)

        #Predict probabilites, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob

import warnings
warnings.filterwarnings('ignore')

#use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
#print pred_prob[0]

pred_churn = pred_prob[:,1]
#print pred_churn
is_churn = y == 1
#print is_churn
# number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)
print counts
# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

#pandas-fu
counts = pd.concat([counts, true_prob],axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts
#print counts






