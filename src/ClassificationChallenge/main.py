
import numpy as np
from numpy import genfromtxt
from sklearn import decomposition, tree
import seaborn as sns

from labfuns import *
from lab3 import BoostClassifier, BayesClassifier


# 1) Import
# 2) Classification via Decisiontree
# 3) Classificaition via Boosted Decisiontree

# 1) Import

# 2) Classification via Decisiontree
# print("Test Challenge Data Bayes")
# testClassifier(BayesClassifier(), dataset='challenge', split=0.7)
# >> Very bad results... 7.41 % accuracy


from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=13, n_jobs=-1)

print("Test Challenge Data Decisiontree")
testClassifier(rnd_clf, dataset='challenge', split=0.7)
# Trial: 0 Accuracy 79.8
# Trial: 10 Accuracy 80.7
# Trial: 20 Accuracy 81.6
# Trial: 30 Accuracy 83.4
# Trial: 40 Accuracy 79.4
# Trial: 50 Accuracy 81.6
# Trial: 60 Accuracy 82.5
# Trial: 70 Accuracy 83.4
# Trial: 80 Accuracy 83.9
# Trial: 90 Accuracy 81.2
# Final mean classification accuracy  81.8 with standard deviation 2.08

print("Evaluation Data set with Decisiontree Classification")
evaluateClassifier(rnd_clf, labellist=['Boborg','Jorgsuto','Atsutobob'], dataset='challengetest')


# test1 = DecisionTreeClassifier()
# print("Test Challenge Data Decisiontree")
# testClassifier(test1, dataset='challenge', split=0.7)
# # >> 73.3 % accuracy, Good.

# print("Evaluation Data set with Decisiontree Classification")
# evaluateClassifier(test1, labellist=['Boborg','Jorgsuto','Atsutobob'], dataset='challengetest')

# print("Test Challenge Data Decisiontree")
# testClassifier(SVMClassifier(), dataset='challenge', split=0.7)
# >> 'linear'-Kernel: 46 % accuracy. Low
# >> 'rbf'-Kernel: 42.3 % accuracy. once 70.3 % -> mean: 51.4
# >> 'poly'-Kernel: 42.3 % accuracy. once 65.3 % -> mean: 49.3
# >> 'sigmoid'-Kernel: 42.3 % accuracy. once 57 % -> mean: 47.2 %

# print("Test Challenge Data Boosted Decisiontree")
# testClassifier(BoostClassifier(SVMClassifier(),T=10), dataset='challenge', split=0.7)
# >> Not sucessful

# print("Test Challenge Data Boosted Decisiontree")
# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='challenge', split=0.7)