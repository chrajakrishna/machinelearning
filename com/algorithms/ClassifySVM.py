'''
Created on Jun 25, 2017

@author: RajaKrishna
'''
from dask.array.random import gamma
def classifySVC(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
        
    ### your code goes here!
    from sklearn import svm
    clf = svm.SVC(kernel="rbf",gamma = 2.0, C= 10.0)
    clf.fit(features_train, labels_train)
    return clf

def classifyDT(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier
    
        
    ### your code goes here!
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=2)
    clf.fit(features_train, labels_train)
    return clf