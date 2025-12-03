#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 20:54:14 2025

@author: Quentin
"""

# Importing necessary libraries
import pandas as pd # Pandas for data reading and manipulation
import numpy as np # Numpy for mathematical calculation and manipulating arrays
from sklearn import metrics # Scikit Learn metrics for computing confusion matrices and accuracy scores
from sklearn import model_selection # Scikit Learn model_selection to apply K-Fold cross validation
import matplotlib.pyplot as plt # matplotlib to display images
import datetime # datetime to create timestamps
from sklearn import linear_model # Scikit Learn linear model for creating a Perceptron
from sklearn import tree # Scikit Learn tree for creating a decision tree
from sklearn import neighbors # Scikit Learn neighbors for creating a KNN
from sklearn import svm # Scikit Learn svm for creating a support vector machine


###############################################################################
# Task1 - Pre-processing and visualisation
# Define a function to preprocess the data and provide a visual of an object of each label
def preprocess(): 
    # Read the data from the csv file
    df = pd.read_csv('/Users/Quentin/Desktop/Hdip Data Science and Analytics/Year 1/Semester 3/COMP8043 - Machine Learning/Assignment 2/fashion-mnist_train.csv')
    # Filter for specific labels
    data = df[df['label'].isin([5,7,9])]
    
    # Make labels and feature subset
    labels = data['label']
    features = data.drop('label', axis=1)
    
    # Declare lists of articles and article type
    article_list = []
    article_titles = []
    
    # For each unique labels
    for label in labels.unique():
        # Associate correct title
        if label == 9:
            article_titles.append('Ankle boot')
        elif label == 5:
            article_titles.append('Sandal')
        elif label == 7:
            article_titles.append('Sneakers')
        # get the indexes of the articl type
        article_indexes = labels[labels == label].index
        # Extract the features at correct indexes
        article_features = features.loc[article_indexes]
        # Get first article features
        first_article_pixels = article_features.loc[article_indexes[0]]
        # append the features to the article list
        article_list.append(first_article_pixels)
    
    # For each article in the list
    i=0
    for article in article_list:
        # Create a plot and display the article image
        plt.figure()
        plt.title(article_titles[i])
        plt.imshow(article.to_numpy().reshape(28,28))
        i+=1
        
    # Return pre-processed features and labels
    return features,labels


###############################################################################
# Task2 - evaluation procedure
# Create a k-fold evaluation procedure to be used by each classifier
def evaluate(features,labels,n,clf):
    
    # Compute Numpy arrays for features and labels
    labels=labels.to_numpy()
    features = features.to_numpy()
    
    # Declare lists for training and predictions durations and accuracies
    all_train_durations = []
    all_pred_durations = []
    allResults = []
    
    
    # Split set into training set (80% of data) and test set (20% of data) with n sample of data
    test_features, train_features, test_labels, train_labels = model_selection.train_test_split(features[:n],
                                                                                                labels[:n],
                                                                                                test_size=0.8,
                                                                                                random_state=0)

    
    # Creating a K-Fold cross validation procedure (Using 5 folds, shuffling indexes, setting random_state as 42 for reproducibility)
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=42)
    
    i = 1
    # For each train and test indexes in the K-fold procedure split
    for train_index, test_index in kf.split(train_features, train_labels):
        # For each split, train the classifier and record training duration
        print('Split: ',i)
        train_start = datetime.datetime.now()
        clf.fit(train_features[train_index], train_labels[train_index])
        train_end = datetime.datetime.now()
        train_duration = (train_end-train_start).total_seconds()
        print("Processing time required for training: ",train_duration)
        all_train_durations.append(train_duration)
        
        # Record prediction duration
        prediction_start = datetime.datetime.now()
        prediction = clf.predict(train_features[test_index])
        prediction_end = datetime.datetime.now()
        prediction_duration = (prediction_end - prediction_start).total_seconds()
        print('processing time required for prediction: ',prediction_duration)
        all_pred_durations.append(prediction_duration)
        
        # Compute a confusion matrix and accuracy score
        print('confusion matrix: ',metrics.confusion_matrix(train_labels[test_index], prediction))
        # print('accuracy score of the classification: ',metrics.accuracy_score(train_labels[test_index], prediction))
        allResults.append(metrics.accuracy_score(train_labels[test_index], prediction))
        i+=1
    
    # Compute minimum, maximum and average training duration
    min_train_duration = min(all_train_durations)
    print('minimum training time per training sample: ',min_train_duration)
    max_train_duration = max(all_train_durations)
    print('maximum training time per training sample: ',max_train_duration)
    mean_train_duration = np.mean(all_train_durations)
    print('average training time per training sample: ',mean_train_duration)
    
    # Compute minimum, maximum and average prediction duration
    min_pred_duration = min(all_pred_durations)
    print('minimum prediction time per evaluation sample: ',min_pred_duration)
    max_pred_duration = max(all_pred_durations)
    print('maximum prediction time per evaluation sample: ',max_pred_duration)
    mean_pred_duration = np.mean(all_pred_durations)
    print('average prediction time per evaluation sample: ',mean_pred_duration)
    
    # Compute minimum, maximum and mean accuracies
    min_accuracy = min(allResults)
    print('minimum prediction accuracy: ',min_accuracy)
    max_accuracy = max(allResults)
    print('maximum prediction accuracy: ',max_accuracy)
    mean_accuracy = np.mean(allResults)
    print('average prediction accuracy: ',mean_accuracy)
    
    # Compute a runtime using the whole set
    start_runtime = datetime.datetime.now()
    clf.fit(train_features, train_labels)
    prediction = clf.predict(test_features)
    end_runtime = datetime.datetime.now()
    runtime = (end_runtime - start_runtime).total_seconds()
    
    # Display confusion matrix and accuracy score on final evaluation
    print('Final confusion matrix: ',metrics.confusion_matrix(test_labels, prediction))
    final_accuracy = metrics.accuracy_score(test_labels, prediction)
    print('Final accuracy score of the classification: ',metrics.accuracy_score(test_labels, prediction))
    print('Classifier runtime: ',runtime)
    
    # Return metrics computed
    return min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy
    

###############################################################################
# Task3 - Perceptron
# Define a function to train and evaluate a Perceptron classifer
def perceptron_classify(features,labels):
    # Instantiate a Perceptron
    clf = linear_model.Perceptron()
    
    # Make an evaluation of the perceptron and display the mean accuracy
    min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,len(features),clf)
    print("Perceptron Mean Accuracy: ",mean_accuracy)
    
    # Declare lists to store sample sizes and metrics
    sample_sizes = []
    min_train_durations = []
    max_train_durations = []
    mean_train_durations = []
    min_pred_durations = []
    max_pred_durations = []
    mean_pred_durations = []
    runtimes = []
    min_accuracies = []
    max_accuracies = []
    mean_accuracies = []
    final_accuracies = []
    
    
    
    # for 20 different sample sizes
    for size in range(1 ,21):
        # Increase sample size by 1/20 and store it
        sample_size = (len(features)//20)*size
        sample_sizes.append(sample_size)
        
        # Evaluate the perceptron with different size and store metrics
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,sample_size,clf)
        min_train_durations.append(min_train_duration)
        max_train_durations.append(max_train_duration)
        mean_train_durations.append(mean_train_duration)
        min_pred_durations.append(min_pred_duration)
        max_pred_durations.append(max_pred_duration)
        mean_pred_durations.append(mean_pred_duration)
        runtimes.append(runtime)
        min_accuracies.append(min_accuracy)
        max_accuracies.append(max_accuracy)
        mean_accuracies.append(mean_accuracy)
        final_accuracies.append(final_accuracy)
    
    # Display a lineplot of the runtimes for each sample sizes
    plt.figure()
    plt.title('Perceptron classification runtimes against sample sizes')
    plt.xlabel('Sample size')
    plt.ylabel('Runtime (seconds)')
    plt.plot(sample_sizes, runtimes,'o-')
    plt.plot(sample_sizes, mean_train_durations,'o-')
    plt.plot(sample_sizes, mean_pred_durations,'o-')
    plt.legend(["Whole runtime", "Mean Training duration",'Mean Prediction duration'], loc="upper left")
    plt.show()
    
    # Create a dataframe of the metrics against sample sizes
    perceptron_df = pd.DataFrame({'sample_sizes': sample_sizes,'min_train_durations': min_train_durations,'max_train_durations': max_train_durations,
                                  'mean_train_durations': mean_train_durations,'min_pred_durations': min_pred_durations,'max_pred_durations': max_pred_durations,
                                  'mean_pred_durations': mean_pred_durations,'runtimes': runtimes,'min_accuracies': min_accuracies,
                                  'max_accuracies': max_accuracies,'mean_accuracies': mean_accuracies,'final_accuracies': final_accuracies})
    
    # Output the datain a a CSV file
    perceptron_df.to_csv('Perceptron.csv', index=False)
    
###############################################################################
# Task4 - Decision trees
# Define a function to train and evaluate a Decision Tree classifer
def tree_classify(features,labels):
    # Instantiate a Decision Tree
    clf = tree.DecisionTreeClassifier()
    
    # Make an evaluation of the Decision tree and display the mean accuracy
    min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,len(features),clf)
    print("Decision Tree Mean Accuracy: ",mean_accuracy)
    
    # Declare lists to store sample sizes and metrics
    sample_sizes = []
    min_train_durations = []
    max_train_durations = []
    mean_train_durations = []
    min_pred_durations = []
    max_pred_durations = []
    mean_pred_durations = []
    runtimes = []
    min_accuracies = []
    max_accuracies = []
    mean_accuracies = []
    final_accuracies = []
    
    
    # for 20 different sample sizes
    for size in range(1 ,21):
        # Increase sample size by 1/20 and store it
        sample_size = (len(features)//20)*size
        sample_sizes.append(sample_size)
        # Evaluate the Decision Tree with different size and store metrics
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,sample_size,clf)
        min_train_durations.append(min_train_duration)
        max_train_durations.append(max_train_duration)
        mean_train_durations.append(mean_train_duration)
        min_pred_durations.append(min_pred_duration)
        max_pred_durations.append(max_pred_duration)
        mean_pred_durations.append(mean_pred_duration)
        runtimes.append(runtime)
        min_accuracies.append(min_accuracy)
        max_accuracies.append(max_accuracy)
        mean_accuracies.append(mean_accuracy)
        final_accuracies.append(final_accuracy)
    
    # Display a lineplot of the runtimes for each sample sizes
    plt.figure()
    plt.title('Decision Tree classification runtimes against sample sizes')
    plt.xlabel('Sample size')
    plt.ylabel('Runtime (seconds)')
    plt.plot(sample_sizes, runtimes,'o-')
    plt.plot(sample_sizes, mean_train_durations,'o-')
    plt.plot(sample_sizes, mean_pred_durations,'o-')
    plt.legend(["Whole runtime", "Mean Training duration",'Mean Prediction duration'], loc="upper left")
    plt.show()
    
    # Create a dataframe of the metrics against sample sizes
    decision_tree_df = pd.DataFrame({'sample_sizes': sample_sizes,'min_train_durations': min_train_durations,'max_train_durations': max_train_durations,
                                  'mean_train_durations': mean_train_durations,'min_pred_durations': min_pred_durations,'max_pred_durations': max_pred_durations,
                                  'mean_pred_durations': mean_pred_durations,'runtimes': runtimes,'min_accuracies': min_accuracies,
                                  'max_accuracies': max_accuracies,'mean_accuracies': mean_accuracies,'final_accuracies': final_accuracies})
    
    # Output the datain a a CSV file
    decision_tree_df.to_csv('Decision Tree.csv', index=False)

###############################################################################
# Task5 - k-nearest Neighbours
# Define a function to train and evaluate a KNN classifer
def knn_classify(features,labels):
    
    # Declare list for accuracies and parameter
    knn_accuracies =[]
    n_neighbors = []
    
    # for each parameter between 1 and 9
    for n_neighbor in range(1,10):
        # Store k parameter
        n_neighbors.append(n_neighbor)
        # train the classifier on this parameter
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbor, metric='minkowski')
        # Evaluate the classifier with the k parameter and store the accuracy
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,len(features),clf)
        knn_accuracies.append(mean_accuracy)
    
    # Display best accuracy
    print('best mean accuracy: ',max(knn_accuracies))
    
    # Select parameter with best accuracy
    best_k_param = n_neighbors[knn_accuracies.index(max(knn_accuracies))]
    print('best k parameter: ', best_k_param)
    
    # Declare lists to store sample sizes and metrics
    sample_sizes = []
    min_train_durations = []
    max_train_durations = []
    mean_train_durations = []
    min_pred_durations = []
    max_pred_durations = []
    mean_pred_durations = []
    runtimes = []
    min_accuracies = []
    max_accuracies = []
    mean_accuracies = []
    final_accuracies = []

    # Train the KNN classifier with best parameter
    clf = neighbors.KNeighborsClassifier(n_neighbors=best_k_param, metric='minkowski')
    
    # for 20 different sample sizes
    for size in range(1 ,21):
        # Increase sample size by 1/20 and store it
        sample_size = (len(features)//20)*size
        sample_sizes.append(sample_size)
        # Evaluate the Decision Tree with different size and store metrics
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,sample_size,clf)
        min_train_durations.append(min_train_duration)
        max_train_durations.append(max_train_duration)
        mean_train_durations.append(mean_train_duration)
        min_pred_durations.append(min_pred_duration)
        max_pred_durations.append(max_pred_duration)
        mean_pred_durations.append(mean_pred_duration)
        runtimes.append(runtime)
        min_accuracies.append(min_accuracy)
        max_accuracies.append(max_accuracy)
        mean_accuracies.append(mean_accuracy)
        final_accuracies.append(final_accuracy)
    
    # Display a lineplot of the runtimes for each sample sizes
    plt.figure()
    plt.title('KNN classification runtimes against sample sizes')
    plt.xlabel('Sample size')
    plt.ylabel('Runtime (seconds)')
    plt.plot(sample_sizes, runtimes,'o-')
    plt.plot(sample_sizes, mean_train_durations,'o-')
    plt.plot(sample_sizes, mean_pred_durations,'o-')
    plt.legend(["Whole runtime", "Mean Training duration",'Mean Prediction duration'], loc="upper left")
    plt.show()
    
    # Create a dataframe of the metrics against sample sizes
    knn_df = pd.DataFrame({'sample_sizes': sample_sizes,'min_train_durations': min_train_durations,'max_train_durations': max_train_durations,
                                  'mean_train_durations': mean_train_durations,'min_pred_durations': min_pred_durations,'max_pred_durations': max_pred_durations,
                                  'mean_pred_durations': mean_pred_durations,'runtimes': runtimes,'min_accuracies': min_accuracies,
                                  'max_accuracies': max_accuracies,'mean_accuracies': mean_accuracies,'final_accuracies': final_accuracies})
    
    # Output the datain a a CSV file
    knn_df.to_csv('KNN.csv', index=False)
###############################################################################
# Task6 - Support Vector Machine
# Define a function to train and evaluate a SVM classifer
def svm_classify(features,labels):
    
    # Declare list for accuracies and parameter
    svm_accuracies =[]
    gammas = []
    
    # for each parameter between -8 and 0
    for gamma in range(-8,0):
        # Store gamma parameter
        gammas.append(gamma)
        # train the classifier on this parameter (10^-gamma)
        clf = svm.SVC(gamma=float(f'1e{gamma}'))
        # Evaluate the classifier with the gamma parameter and store the accuracy
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,len(features),clf)
        svm_accuracies.append(mean_accuracy)
    
    # Display best accuracy
    print('best mean accuracy: ',max(svm_accuracies))
    
    # Select parameter with best accuracy
    best_gamma_param = gammas[svm_accuracies.index(max(svm_accuracies))]
    print('best gamma parameter: ', best_gamma_param)
    
    # Declare lists to store sample sizes and metrics
    sample_sizes = []
    min_train_durations = []
    max_train_durations = []
    mean_train_durations = []
    min_pred_durations = []
    max_pred_durations = []
    mean_pred_durations = []
    runtimes = []
    min_accuracies = []
    max_accuracies = []
    mean_accuracies = []
    final_accuracies = []
    
    # Train the SVM classifier with best parameter
    clf = svm.SVC(gamma=float(f'1e{best_gamma_param}'))
    
    # for 20 different sample sizes
    for size in range(1 ,21):
        # Increase sample size by 1/20 and store it
        sample_size = (len(features)//20)*size
        sample_sizes.append(sample_size)
        # Evaluate the Decision Tree with different size and store metrics
        min_train_duration,max_train_duration,mean_train_duration,min_pred_duration,max_pred_duration,mean_pred_duration,min_accuracy,max_accuracy,mean_accuracy,runtime,final_accuracy = evaluate(features,labels,sample_size,clf)
        min_train_durations.append(min_train_duration)
        max_train_durations.append(max_train_duration)
        mean_train_durations.append(mean_train_duration)
        min_pred_durations.append(min_pred_duration)
        max_pred_durations.append(max_pred_duration)
        mean_pred_durations.append(mean_pred_duration)
        runtimes.append(runtime)
        min_accuracies.append(min_accuracy)
        max_accuracies.append(max_accuracy)
        mean_accuracies.append(mean_accuracy)
        final_accuracies.append(final_accuracy)
    
    # Display a lineplot of the runtimes for each sample sizes
    plt.figure()
    plt.title('SVM classification runtimes against sample sizes')
    plt.xlabel('Sample size')
    plt.ylabel('Runtime (seconds)')
    plt.plot(sample_sizes, runtimes,'o-')
    plt.plot(sample_sizes, mean_train_durations,'o-')
    plt.plot(sample_sizes, mean_pred_durations,'o-')
    plt.legend(["Whole runtime", "Mean Training duration",'Mean Prediction duration'], loc="upper left")
    plt.show()

    # Create a dataframe of the metrics against sample sizes
    svm_df = pd.DataFrame({'sample_sizes': sample_sizes,'min_train_durations': min_train_durations,'max_train_durations': max_train_durations,
                                  'mean_train_durations': mean_train_durations,'min_pred_durations': min_pred_durations,'max_pred_durations': max_pred_durations,
                                  'mean_pred_durations': mean_pred_durations,'runtimes': runtimes,'min_accuracies': min_accuracies,
                                  'max_accuracies': max_accuracies,'mean_accuracies': mean_accuracies,'final_accuracies': final_accuracies})
    
    # Output the datain a a CSV file
    svm_df.to_csv('SVM.csv', index=False)
    
    print('best mean accuracy: ',max(svm_accuracies))
    
    best_gamma_param = gammas[svm_accuracies.index(max(svm_accuracies))]
    print('best gamma parameter: ', best_gamma_param)

###############################################################################
# Task7 - Comparison
# Define main function to run
def main():
    # Preprocess data and evaluate each models (comment out models not to be evaluated)
    features,labels = preprocess()
    perceptron_classify(features,labels)
    tree_classify(features,labels)
    knn_classify(features,labels)
    svm_classify(features,labels)

# Executing main function
main()