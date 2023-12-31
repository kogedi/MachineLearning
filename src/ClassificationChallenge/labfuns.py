from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ColorConverter
import random as rnd
from sklearn.datasets import make_blobs #samples_generator import make_blobs
from sklearn import decomposition, tree, svm
import os
import getpass
import csv
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris  # Example dataset

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import seaborn as sns
sns.set()



def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    ellip.set_alpha(0.25)

    ax.add_artist(ellip)
    return ellip


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data.
def trteSplit(X,y,pcSplit,seed=None):
    # Compute split indices
    Ndata = X.shape[0]
    Ntr = int(np.rint(Ndata*pcSplit))
    Nte = Ndata-Ntr
    np.random.seed(seed)    
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    xTr = X[trIdx,:]
    yTr = y[trIdx]
    xTe = X[teIdx,:]
    yTe = y[teIdx]
    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    """Splits data into training and test set

    Args:
        X (_type_): Data features
        y (_type_): labels
        pcSplit (float [0, 1]): defines the percent of the data should be used as training data
        seed (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx

# def equalweigtData(X,y):
#     """Splits data into training and test set

#     Args:
#         X (_type_): Data features
#         y (_type_): labels
#         pcSplit (float [0, 1]): defines the percent of the data should be used as training data
#         seed (_type_, optional): _description_. Defaults to None.

#     Returns:
#         _type_: _description_
#     """
#     labels = np.unique(y)
#     xTr = np.zeros((0,X.shape[1]))
    
#     trIdx = np.zeros((0,),dtype=int)
    
#     for label in labels:
#         classIdx = np.where(y==label)[0]
#         NPerClass = len(classIdx)
#         Ntr = int(np.rint(NPerClass*pcSplit))
#         idx = np.random.permutation(NPerClass)
#         trClIdx = classIdx[idx[:Ntr]]
#         teClIdx = classIdx[idx[Ntr:]]
#         trIdx = np.hstack((trIdx,trClIdx))
#         teIdx = np.hstack((teIdx,teClIdx))
#         # Split data
#         xTr = np.vstack((xTr,X[trClIdx,:]))
#         yTr = np.hstack((yTr,y[trClIdx]))
        

#     return xTr,yTr

def balance_dataset(X, y):
    unique_classes, class_counts = np.unique(y, return_counts=True)

    # Find the class with the maximum samples
    majority_class = unique_classes[np.argmax(class_counts)]
    majority_class_count = np.max(class_counts)

    # Identify minority classes
    minority_classes = unique_classes[class_counts < majority_class_count]

    # Oversample minority classes
    for minority_class in minority_classes:
        minority_indices = np.where(y == minority_class)[0]
        oversampled_indices = np.random.choice(minority_indices, size=majority_class_count - len(minority_indices), replace=True)
        X_oversampled = X[oversampled_indices]
        y_oversampled = y[oversampled_indices]

        X = np.vstack((X, X_oversampled))
        y = np.concatenate((y, y_oversampled))

    # Shuffle the dataset
    shuffle_indices = np.random.permutation(len(X))
    X = X[shuffle_indices]
    y = y[shuffle_indices]

    return X, y

def fetchEvalDataset(dataset='challengetest'):
    
    if dataset == 'challengetest':
        try:
            X = genfromtxt('C:/Users/Konra/git_repos//MachineLearning/src/ClassificationChallenge/EvaluateOnMe2.csv', delimiter=',') # ChalXf2
        except FileNotFoundError:
            print("Challengetest-File not found!")   
    else:
            print("Please specify a dataset!")
            X = np.zeros(0)
            
    return X

def fetchDataset(dataset='iris'):
        

    # Get the username of the currently logged-in user
    username = getpass.getuser()

    #print("Username:", username)
    if username == 'Konrad Dittrich':
        
        if dataset == 'challenge':
            try:
                y = genfromtxt('C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/ChalYf2.csv', delimiter=',')
                X = genfromtxt('C:/Users/Konrad Dittrich/git/repos/MachineLearning/src/ClassificationChallenge/ChalXf2.csv', delimiter=',')
                
                pcadim = 12
            except FileNotFoundError:
                print("Challenge File not found!")
                      
        
        elif dataset == 'iris':
            try:
                X = genfromtxt('.\src\Bayes_boosting\lab3py\irisX.txt', delimiter=',')
                y = genfromtxt('.\src\Bayes_boosting\lab3py\irisY.txt', delimiter=',')#-1 #,dtype=np.int
                pcadim = 2
            except FileNotFoundError:
                print("iris-File not found!")
        elif dataset == 'wine':
            X = genfromtxt('wineX.txt', delimiter=',')
            y = genfromtxt('wineY.txt', delimiter=',',dtype=np.int)-1
            pcadim = 0
        elif dataset == 'olivetti':
            X = genfromtxt('.\src\Bayes_boosting\lab3py\olivettifacesX.txt', delimiter=',')
            X = X/255
            y = genfromtxt('.\src\Bayes_boosting\lab3py\olivettifacesY.txt', delimiter=',')#,dtype=np.int)
            pcadim = 20
        elif dataset == 'vowel':
            try:
                X = genfromtxt('.\src\Bayes_boosting\lab3py\AvowelX.txt', delimiter=',')
                y = genfromtxt('.\src\Bayes_boosting\lab3py\AvowelY.txt', delimiter=',')#,dtype=np.int)
                pcadim = 0
            except FileNotFoundError:
                print("File not found!")
        else:
            print("Please specify a dataset!")
            X = np.zeros(0)
            y = np.zeros(0)
            pcadim = 0
            
    elif username == 'Konra':
        if dataset == 'challenge':
            try:
                y = genfromtxt('C:/Users/Konra/git_repos/MachineLearning/src/ClassificationChallenge/ChalYf2.csv', delimiter=',')
                X = genfromtxt('C:/Users/Konra/git_repos/MachineLearning/src/ClassificationChallenge/ChalXf2.csv', delimiter=',')
                pcadim = 2
            except FileNotFoundError:
                print("File not found!")
        elif dataset == 'wine':
            X = genfromtxt('wineX.txt', delimiter=',')
            y = genfromtxt('wineY.txt', delimiter=',',dtype=np.int)-1
            pcadim = 0
        elif dataset == 'olivetti':
            X = genfromtxt('./src/Bayes_boosting/lab3py/olivettifacesX.txt', delimiter=',')
            X = X/255
            y = genfromtxt('./src/Bayes_boosting/lab3py/olivettifacesY.txt', delimiter=',')#,dtype=np.int)
            pcadim = 20
        elif dataset == 'vowel':
            try:
                X = genfromtxt('./src/Bayes_boosting/lab3py/AvowelX.txt', delimiter=',')
                y = genfromtxt('./src/Bayes_boosting/lab3py/AvowelY.txt', delimiter=',')#,dtype=np.int)
                pcadim = 0
            except FileNotFoundError:
                print("File not found!")
        else:
            print("Please specify a dataset!")
            X = np.zeros(0)
            y = np.zeros(0)
            pcadim = 0
            
    return X,y,pcadim


def genBlobs(n_samples=200,centers=5,n_features=2):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features,random_state=0)
    return X,y


# Scatter plots the two first dimension of the given data matrix X
# and colors the points by the labels.
def scatter2D(X,y):
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = np.where(y==label)[0]
        Xclass = X[classIdx,:]
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


def plotGaussian(X,y,mu,sigma):
    
    labels = np.unique(y)
    Ncolors = len(labels)
    xx = np.arange(Ncolors)
    ys = [i+xx+(i*xx)**2 for i in range(Ncolors)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    c = 1.0
    for label in labels:
        classIdx = y==label
        Xclass = X[classIdx,:]
        plot_cov_ellipse(sigma[label], mu[label])
        plt.scatter(Xclass[:,0],Xclass[:,1],linewidths=1,s=25,color=colors[label],marker='o',alpha=0.75)
        c += 1.

    plt.show()


# The function below, `testClassifier`, will be used to try out the different datasets.
# `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`.
# Observe that we split the data into a **training** and a **testing** set.
def testClassifier(classifier, usepredict, NN, dataset='iris', dim=0, split=0.7, ntrials=100, filterrange=2): # 1.33
    """
    A Test function for different Classifiers.
    
    Fetchs input data, drops outliers, plots the scatter matrix, splits data for training and test, 
    performs PCA dimensionality reduction, trains a classifier and predicts output.

    Args:
    ------
        classifier (classifier object): instance of the choosen classifier class
        dataset (str, optional): dataset for fetchData, name has to be defined. Defaults to 'iris'.
        dim (int, optional): dimension of the inputdata. Defaults to 0.
        split (float, optional): Split ratio of training and test data. Defaults to 0.7.
        ntrials (int, optional): number of trails to gain statistical certainty about accuracy. Defaults to 100.
    
    Examples 
    --------
    >>> trained_clf = testClassifier(rnd_clf, dataset='challenge', split=0.7)
    """
    X,y,pcadim = fetchDataset(dataset)
    
    #Pandas variables
    Xval = pd.DataFrame(X)
    yval = pd.DataFrame(y)
    
    #outlaier detection
    # Initialize empty arrays for lower and upper bounds for each feature
    lower_bounds = np.zeros(len(Xval.columns))
    upper_bounds = np.zeros(len(Xval.columns))

    # Calculate lower and upper bounds for each feature based on IQR
    for i, feature_name in enumerate(Xval.columns):
        feature_data = Xval[feature_name]  # Extract the data for the current feature
        Q1 = np.percentile(feature_data, 25)  # Calculate the 25th percentile
        Q3 = np.percentile(feature_data, 75)  # Calculate the 75th percentile
        IQR = Q3 - Q1  # Calculate the interquartile range

        # Define lower and upper bounds for the current feature
        lower_bounds[i] = Q1 - filterrange * IQR
        upper_bounds[i] = Q3 + filterrange * IQR

    # Print lower and upper bounds for each feature
    for i, feature_name in enumerate(Xval.columns):
        print(f'Feature: {feature_name}, Lower Bound: {lower_bounds[i]}, Upper Bound: {upper_bounds[i]}')
    
    # Exclude outliers from the data for each feature
    filtered_Xdata = Xval[
        (Xval >= lower_bounds) & (Xval <= upper_bounds)
    ].dropna()
    # Get the indices of rows that were not filtered out
    indices_to_keep = filtered_Xdata.index
    # Filter yval based on the same indices to keep labels consistent
    filtered_Ydata = yval.loc[indices_to_keep]

    
    scatter_matrix(filtered_Xdata,figsize=(6,4), diagonal='kde', c=filtered_Ydata, cmap='viridis')
    plt.show()  
    
   
    # Reshape for the old code of lab3
    X =  filtered_Xdata.values
    y =  filtered_Ydata.values.reshape(-1)
    
    
    means = np.zeros(ntrials,) # mean for accuracy calculation
    X, y = balance_dataset(X,y)
    
    for trial in range(ntrials):
        
        
        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)
        


        # Do PCA replace default value if user provides it
        # if dim > 0:
        #     pcadim = dim

        # if pcadim > 0:
        #     pca = decomposition.PCA(n_components=pcadim)
        #     pca.fit(xTr)
        #     xTr = pca.transform(xTr)
        #     xTe = pca.transform(xTe)
        if NN == True:

            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(classifier.parameters(), lr=0.01)

            # Convert your data to PyTorch tensors
            # Assuming you have training_data and labels as your data
            # Make sure to convert them to torch tensors before training

            # Training loop
            num_epochs = 1000  # You can adjust the number of epochs as needed

            for epoch in range(num_epochs):
                # Forward pass
                x_in = torch.tensor((xTr.shape[1]), dtype=torch.float32)
                x_in = torch.from_numpy(np.float32(xTr))
                outputs = classifier(x_in)
                
                y_in = torch.tensor((yTr.shape[0]), dtype=torch.int)
                y_in = torch.from_numpy(np.int64(yTr))
                loss = criterion(outputs, y_in)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print the loss at every epoch
                #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

            # After training, you can use the model for predictions
            # For example, if you have a test_data tensor:
            x_ein = torch.tensor((xTe.shape[1]), dtype=torch.float32)
            x_ein = torch.from_numpy(np.float32(xTe))
            test_outputs = classifier(x_ein)
            yPr_tensor = torch.argmax(test_outputs, dim=1)
            yPr = np.array(yPr_tensor)
        else:
            if usepredict == True:  
                trained_classifier = classifier.fit(xTr, yTr)
                yPr = trained_classifier.predict(xTe)
              
            else:           
                # Train
                trained_classifier = classifier.trainClassifier(xTr, yTr)
                # Predict
                yPr = trained_classifier.classify(xTe)
                pass

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))
    
    return trained_classifier
    
    
def evaluateClassifier(trained_classifier,usepredict, labellist,dataset='challengetest'):

    x = fetchEvalDataset(dataset)
    
    # Predict
    if usepredict == True:
        yPr = trained_classifier.predict(x)
    else:
       yPr = trained_classifier.classify(x)

    print("The labels are: ")
    print(yPr)
    output_file_path = "C:/Users/Konra/git_repos/MachineLearning/src/ClassificationChallenge/classification.csv"

    with open(output_file_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # Writing each element as a separate row
        for element in yPr:
            csvwriter.writerow([labellist[int(element)]])
        
    return yPr

# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.
def plotBoundary(classifier, dataset='iris', split=0.7):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,1)
    classes = np.unique(y)

    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)

    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    # Train
    trained_classifier = classifier.trainClassifier(xTr, yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            # Predict
            grid[yi,xi] = trained_classifier.classify(np.array([[xx, yy]]))

    
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    fig = plt.figure()
    # plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        trClIdx = np.where(y[trIdx] == c)[0]
        teClIdx = np.where(y[teIdx] == c)[0]
        plt.scatter(xTr[trClIdx, 0], xTr[trClIdx, 1], marker='o', color=color, s=40, alpha=0.5, label="Class " + str(c) + " Train")
        plt.scatter(xTe[teClIdx, 0], xTe[teClIdx, 1], marker='*', color=color, s=50, alpha=0.8, label="Class " + str(c) + " Test")
        #plt.scatter(xTr[trClIdx,0],xTr[trClIdx,1],marker='o',c=color,s=40,alpha=0.5, label="Class "+str(c)+" Train")
        #plt.scatter(xTe[teClIdx,0],xTe[teClIdx,1],marker='*',c=color,s=50,alpha=0.8, label="Class "+str(c)+" Test")
    plt.legend(bbox_to_anchor=(1., 1), loc=2, borderaxespad=0.)
    fig.subplots_adjust(right=0.7)
    plt.show()


def visualizeOlivettiVectors(xTr, Xte):
    N = xTr.shape[0]
    Xte = Xte.reshape(64, 64).transpose()
    plt.subplot(1, 2, 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Test image")
    plt.imshow(Xte, cmap=plt.get_cmap('gray'))
    for i in range(0, N):
        plt.subplot(N, 2, 2+2*i)
        plt.xticks([])
        plt.yticks([])
        plt.title("Matched class training image %i" % (i+1))
        X = xTr[i, :].reshape(64, 64).transpose()
        plt.imshow(X, cmap=plt.get_cmap('gray'))
    plt.show()


class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False
        self.classifier = None

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth= int(Xtr.shape[1]/2+1))
        self.classifier = rtn.classifier
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)
    
class SVMClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = SVMClassifier()
        rtn.classifier = svm.SVC(kernel='sigmoid')
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)
    
# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.input_size        = 11
        self.output_size       = 3
        self.n_hidden_layers   = 5
        self.n_neurons         = 128

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(self.input_size, self.n_neurons)
        self.input_layer_activation = nn.ReLU()

        # Create hidden layers with ReLU activation
        self.hidden_layers = nn.ModuleList()
        for _ in range(self.n_hidden_layers):
            self.hidden_layers.append(nn.Linear(self.n_neurons, self.n_neurons))
            self.hidden_layers.append(nn.ReLU())

        # Create output layer
        self.output_layer = nn.Linear(self.n_neurons, self.output_size)


    def forward(self, input: torch.Tensor):
        ''' Performs a forward computation '''

        # Compute first layer
        l_out = self.input_layer(input)
        l_out = self.input_layer_activation(l_out)

        # Compute hidden layers
        for layer in self.hidden_layers:
            l_out = layer(l_out)

        # Compute output layer
        out = self.output_layer(l_out)
        return out
