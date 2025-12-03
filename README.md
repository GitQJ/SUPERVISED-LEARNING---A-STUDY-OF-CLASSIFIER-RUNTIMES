# SUPERVISED-LEARNING---A-STUDY-OF-CLASSIFIER-RUNTIMES
Studying different Classifiers on MNIST dataset

# Summary
The file “fashion-mnist_train.csv” contains down-sampled product images from Zalando.com.
Every row consists of a label and 28x28 8-bit grayscale pixel values of the product image. The goal of this assignment is to evaluate and optimise the performance of different classifiers for their suitability to classify this dataset.

We are testing our classifier on the data of Ankle boots, Sneakers and Sandals. We are preprocessing the data so that it contains only these classes. We then have created a 5-fold cross-validation procedure to have more consistency on accuracies and runtimes. We are then testing each classifiers with different fractions of the preprocessed sample to observe their pattern on runtimes and also implemented hyper-parameter tuning processes before testing the K-nearest neighbour and the Support Vector Machine.

The Perceptron is not the worst classifier with a mean accuracy of 0.91. Our study seems to indicate being the quickest, however we observed a variance in terms of runtime and training, as convergence is not reach solely based on sample size (number of iteration changing).

The Decision Tree is the less accurate with a mean accuracy of 0.89 (the worst of the comparison). It is also the second slowest in terms of runtime and training times due to the node computation but it is the quickest to make predictions.

The KNN performs reasonably well with a mean accuracy of 0.93. It is also very quick at training and predicts generally in less than a second. We found out the best hyper-parameter as 2 nearest neighbours.

We can see that SVM is the most accurate with a mean accuracy of 0.96 but it is at the significant trade of time because it is generally the slowest in general (runtime, training, predicting). We found out the best hyper-parameter as gamma parameter 1e-6.

If runtime is not to be considered, we would recommend the SVM as best classifier as it captures the non-linear relationships in the dataset, and performs very well on unseen data.

If runtime is to be considered, we can recommend the KNN as second best classifier as it is a relatively simple and intuitive algorithm to train and use. It generally runs in about a second and onboards training data instantly.

# Tools
- Python
- Pandas
- Numpy
- Sci-Kit Learn
- Datetime
