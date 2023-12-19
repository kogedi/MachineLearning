
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

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


#************OUTLINE****************************
# 0) Classification via NeuralNetwork
# 1) Classification via XGBoostClassifier
# 2) Classification via RandomForestClassifier 
# 3) Classification via Decisiontree
# 4) Classification via BayesClassifier
# 5) Classification via Support Vector Machine 
# 6) Classification via Boosted Decisiontree
# 7) Classification via Boosted Support Vector Machine 
#************************************************

# TODO: uncomment one of the following lines
# TODO: input your own directory in labfuns.py fetchDataset() in testClassifier()
# TODO: preprocessing: Use help functions to preprocess data
# e.g. 
# Nymble : 1.0
# ÖstraStationen : 2.0
# SisterOchBro : 3.0
# 7-11 : 4.0
# Q : 5.0
# True : 1
# False : 2
# y
# Jorgutob : 0
# Atsorg : 1
# Bobsuto : 2
# TODO: Change labellist depending on your labels, or generalize
# TODO: Change parameters of classifiers
# TODO: uncomment scatter matrix for increase performance, use scattermatrix for corelation detection

# best solution
#classificationlist = ['XGBoostClassifier']
#good classifiers
#classificationlist = ['XGBoostClassifier','RandomForestClassifier','BayesClassifier']
# all classifiers
classificationlist = ['XGBoostClassifier', 'RandomForestClassifier','Decisiontree','BayesClassifier','SupportVectorMachine', 'BoostedDecisiontree', 'BoostedSupportVectorMachine', 'XGBoostClassifier']
#classificationlist = ['NN'] #Draft Classifier

# Parameters
labellist = ['Jorgutob','Atsorg','Bobsuto'] # Jorgutob : 0, Atsorg : 1, Bobsuto : 2
outlier_sigma_range = 2.5 # factor of standard deviation to be allowed. Otherwise outlier are cut of
training_dataset = 'challenge'
interfer_dataset = 'challengetest'
train_test_split = 0.9 # split ratio in training and test dataset

# 0) Classification via NeuralNetwork
    
if 'NN' in classificationlist:
    
    print("NN", )
    # Define the number of input features, hidden units, and output classes
    model = NeuralNetwork(input_size=11, hidden_size=128, output_size=3)
            
    print("Test Challenge NN")
    trained_clf = testClassifier(model, usepredict=True, NN=True, dataset=training_dataset, split=train_test_split, ntrials=20, filterrange=outlier_sigma_range) #3 sehr gut 2.6 auch

    print("Evaluation Data set with NN ")
    evaluateClassifier(trained_clf, usepredict=True, labellist=labellist, dataset=interfer_dataset)

# 1) Classificaition via XGBoostClassifier

if 'XGBoostClassifier' in classificationlist:
    
    print("XGBoostClassifier", )
    clf = xgb.XGBClassifier(objective='multi:softmax', num_class=3)
    
    print("Test Challenge XGBoostClassifier")
    trained_clf = testClassifier(clf, usepredict=True, NN=False, dataset=training_dataset, split=train_test_split, ntrials=40, filterrange=outlier_sigma_range)

    print("Evaluation Data set with XGBoostClassifier ")
    evaluateClassifier(trained_clf, usepredict=True, labellist=labellist, dataset=interfer_dataset)
    

# 2) Classificaition via RandomForestClassifier 

if 'RandomForestClassifier' in classificationlist:
    from sklearn.ensemble import RandomForestClassifier

    print('RandomForestClassifier')
    rnd_clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_leaf_nodes=12) #, max_leaf_nodes=13

    print("Test Challenge RandomForestClassifier")
    trained_clf = testClassifier(rnd_clf, usepredict=True, NN=False, dataset=training_dataset, split=train_test_split, ntrials=80, filterrange=outlier_sigma_range)

    print("Evaluation Data set with RandomForestClassifier ")
    evaluateClassifier(trained_clf, usepredict=True, labellist=labellist, dataset=interfer_dataset)


# 3) Classification via Decisiontree

if  'Decisiontree' in classificationlist:
    test1 = DecisionTreeClassifier()
    print("")
    print("Test Challenge DecisionTreeClassifier")
    testClassifier(test1, usepredict=False, NN=False, dataset=training_dataset, split=train_test_split)
    # # >> 73.3 % accuracy, Good.

    print("Evaluation Data set with DecisionTreeClassifier")
    evaluateClassifier(test1, usepredict=False, labellist=labellist, dataset=interfer_dataset)


# 4) Classification via BayesClassifier
if 'BayesClassifier' in classificationlist:
    print("")
    print("Test Challenge BayesClassifier")
    testClassifier(BayesClassifier(), usepredict=False, NN=False, dataset=training_dataset, split=train_test_split)
    # >> Very bad results... 7.41 % accuracy

if 'SupportVectorMachine' in classificationlist:
    # 5) Classification via Support Vector Machine 
    print("")
    print("Test Challenge SVMClassifier")
    testClassifier(SVMClassifier(), usepredict=False, NN=False, dataset=training_dataset, split=train_test_split)
    # >> 'linear'-Kernel: 46 % accuracy. Low
    # >> 'rbf'-Kernel: 42.3 % accuracy. once 70.3 % -> mean: 51.4
    # >> 'poly'-Kernel: 42.3 % accuracy. once 65.3 % -> mean: 49.3
    # >> 'sigmoid'-Kernel: 42.3 % accuracy. once 57 % -> mean: 47.2 %

# 6) Classification via Boosted Decisiontree
if 'BoostedDecisiontree' in classificationlist:
    print("")
    print("Test Challenge Boosted Decisiontree, i")
    testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10 ), NN=False,usepredict=False, dataset=training_dataset, split=train_test_split)
    #>> Not sucessful
    # Final mean classification accuracy  47.6 with standard deviation 1.88


# 7) Classification via Boosted Support Vector Machine 
if 'BoostedSupportVectorMachine' in classificationlist:
    print("")
    print("Test Challenge Boosted SVMClassifier")
    testClassifier(BoostClassifier(SVMClassifier(),T=10), NN=False, usepredict=False, dataset=training_dataset, split=train_test_split)
    #>> Not sucessful
    #Final mean classification accuracy  0 with standard deviation 0
