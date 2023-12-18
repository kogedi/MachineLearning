
import numpy as np
from numpy import genfromtxt
from sklearn import decomposition, tree
import seaborn as sns

from labfuns import *
from lab3 import BoostClassifier, BayesClassifier

#************OUTLINE****************************
# 1) Import
# 2) Classification via RandomForestClassifier 
# 3) Classification via Decisiontree
# 4) Classification via BayesClassifier
# 5) Classification via Support Vector Machine 
# 6) Classification via Boosted Decisiontree
# 7) Classification via Boosted Support Vector Machine 
#************************************************

# TODO: uncomment one of the following lines: 21 for full test, 23 for test for the most sucessfull classifiers
classificationlist = ['RandomForestClassifier','Decisiontree','BayesClassifier','SupportVectorMachine', 'BoostedDecisiontree', 'BoostedSupportVectorMachine']

#classificationlist = ['RandomForestClassifier','Decisiontree']

# 1) Import via fetchDataset() in testClassifier()


# 2) Classificaition via RandomForestClassifier 

if 'RandomForestClassifier' in classificationlist:
    from sklearn.ensemble import RandomForestClassifier

    rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1) #, max_leaf_nodes=13

    print("Test Challenge RandomForestClassifier")
    trained_clf = testClassifier(rnd_clf, dataset='challenge', split=0.7, ntrials=20)

    print("Evaluation Data set with RandomForestClassifier ")
    evaluateClassifier(trained_clf, labellist=['Boborg','Jorgsuto','Atsutobob'], dataset='challengetest')


# 3) Classification via Decisiontree

if  'Decisiontree' in classificationlist:
    test1 = DecisionTreeClassifier()
    print("")
    print("Test Challenge DecisionTreeClassifier")
    testClassifier(test1, dataset='challenge', split=0.7)
    # # >> 73.3 % accuracy, Good.

    print("Evaluation Data set with DecisionTreeClassifier")
    evaluateClassifier(test1, labellist=['Boborg','Jorgsuto','Atsutobob'], dataset='challengetest')


# 4) Classification via BayesClassifier
if 'BayesClassifier' in classificationlist:
    print("")
    print("Test Challenge BayesClassifier")
    testClassifier(BayesClassifier(), dataset='challenge', split=0.7)
    # >> Very bad results... 7.41 % accuracy

if 'SupportVectorMachine' in classificationlist:
    # 5) Classification via Support Vector Machine 
    print("")
    print("Test Challenge SVMClassifier")
    testClassifier(SVMClassifier(), dataset='challenge', split=0.7)
    # >> 'linear'-Kernel: 46 % accuracy. Low
    # >> 'rbf'-Kernel: 42.3 % accuracy. once 70.3 % -> mean: 51.4
    # >> 'poly'-Kernel: 42.3 % accuracy. once 65.3 % -> mean: 49.3
    # >> 'sigmoid'-Kernel: 42.3 % accuracy. once 57 % -> mean: 47.2 %

# 6) Classification via Boosted Decisiontree
if 'BoostedDecisiontree' in classificationlist:
    print("")
    print("Test Challenge Boosted Decisiontree")
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='challenge', split=0.7)
    #>> Not sucessful
    # Final mean classification accuracy  47.6 with standard deviation 1.88


# 7) Classification via Boosted Support Vector Machine 
if 'BoostedSupportVectorMachine' in classificationlist:
    print("")
    print("Test Challenge Boosted SVMClassifier")
    testClassifier(BoostClassifier(SVMClassifier(),T=10), dataset='challenge', split=0.7)
    #>> Not sucessful
    #Final mean classification accuracy  0 with standard deviation 0
