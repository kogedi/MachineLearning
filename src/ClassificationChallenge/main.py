
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

rnd_clf = RandomForestClassifier(n_estimators=1000, n_jobs=-1) #, max_leaf_nodes=13

print("Test Challenge Data Decisiontree")
trained_clf = testClassifier(rnd_clf, dataset='challenge', split=0.7)

print("Evaluation Data set with Decisiontree Classification")
evaluateClassifier(trained_clf, labellist=['Boborg','Jorgsuto','Atsutobob'], dataset='challengetest')






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