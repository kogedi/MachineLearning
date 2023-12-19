
import numpy as np
from numpy import genfromtxt
from sklearn import decomposition, tree
import seaborn as sns

from labfuns import *
from lab3 import BoostClassifier, BayesClassifier

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # Example dataset

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
#classificationlist = ['XGBoostClassifier','RandomForestClassifier']
k = 2
for i in range(2,7):
    classificationlist = [ 'XGBoostClassifier'] #,'RandomForestClassifier','BayesClassifier']

    #classificationlist = ['XGBoostClassifier','RandomForestClassifier','Decisiontree','BayesClassifier','SupportVectorMachine', 'BoostedDecisiontree', 'BoostedSupportVectorMachine']

    # 1) Import via fetchDataset() in testClassifier()

    k = k + 0.2
    # 2) Classificaition via RandomForestClassifier 

    if 'XGBoostClassifier' in classificationlist:
        
        print("XGBoostClassifier", )
        clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
        
        print("Test Challenge XGBoostClassifier", k)
        trained_clf = testClassifier(clf, usepredict=True, dataset='challenge', split=0.9, ntrials=40, filterrange=2.5) #3 sehr gut

        print("Evaluation Data set with XGBoostClassifier ")
        evaluateClassifier(trained_clf, usepredict=True, labellist=['Jorgutob','Atsorg','Bobsuto'], dataset='challengetest')

    if 'RandomForestClassifier' in classificationlist:
        from sklearn.ensemble import RandomForestClassifier

        print('RandomForestClassifier')
        rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_leaf_nodes=12) #, max_leaf_nodes=13

        print("Test Challenge RandomForestClassifier")
        trained_clf = testClassifier(rnd_clf, usepredict=True, dataset='challenge', split=0.9, ntrials=80, filterrange=2) #3

        print("Evaluation Data set with RandomForestClassifier ")
        evaluateClassifier(trained_clf, usepredict=True, labellist=['Jorgutob','Atsorg','Bobsuto'], dataset='challengetest')


    # 3) Classification via Decisiontree

    if  'Decisiontree' in classificationlist:
        test1 = DecisionTreeClassifier()
        print("")
        print("Test Challenge DecisionTreeClassifier")
        testClassifier(test1, usepredict=False, dataset='challenge', split=0.9)
        # # >> 73.3 % accuracy, Good.

        print("Evaluation Data set with DecisionTreeClassifier")
        evaluateClassifier(test1, usepredict=False, labellist=['Jorgutob','Atsorg','Bobsuto'], dataset='challengetest')


    # 4) Classification via BayesClassifier
    if 'BayesClassifier' in classificationlist:
        print("")
        print("Test Challenge BayesClassifier")
        testClassifier(BayesClassifier(), usepredict=False, dataset='challenge', split=0.9)
        # >> Very bad results... 7.41 % accuracy

    if 'SupportVectorMachine' in classificationlist:
        # 5) Classification via Support Vector Machine 
        print("")
        print("Test Challenge SVMClassifier")
        testClassifier(SVMClassifier(), usepredict=False, dataset='challenge', split=0.9)
        # >> 'linear'-Kernel: 46 % accuracy. Low
        # >> 'rbf'-Kernel: 42.3 % accuracy. once 70.3 % -> mean: 51.4
        # >> 'poly'-Kernel: 42.3 % accuracy. once 65.3 % -> mean: 49.3
        # >> 'sigmoid'-Kernel: 42.3 % accuracy. once 57 % -> mean: 47.2 %

    # 6) Classification via Boosted Decisiontree
    if 'BoostedDecisiontree' in classificationlist:
        print("")
        print("Test Challenge Boosted Decisiontree, i", i)
        testClassifier(BoostClassifier(DecisionTreeClassifier(), T=i), usepredict=False, dataset='challenge', split=0.9)
        #>> Not sucessful
        # Final mean classification accuracy  47.6 with standard deviation 1.88


    # 7) Classification via Boosted Support Vector Machine 
    if 'BoostedSupportVectorMachine' in classificationlist:
        print("")
        print("Test Challenge Boosted SVMClassifier")
        testClassifier(BoostClassifier(SVMClassifier(),T=10), usepredict=False, dataset='challenge', split=0.9)
        #>> Not sucessful
        #Final mean classification accuracy  0 with standard deviation 0
