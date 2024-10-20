import numpy as np
from sklearn.metrics import accuracy_score

# 2. Implementation for kernelized Perceptron.
def polykernel(X1, X2, degree=3):
    return (1 + np.dot(X1, X2)) ** degree

# Implementation for training the kernelized Perceptron
def kernelizedPerceptron(X, y, degree=3, max_iter=5):
    # initialize the parameters
    n_samples = X.shape[0]
    n_features = X.shape[1]
    alpha = np.zeros(n_samples)
    sv = X
    sv_labels = y
    classes = np.unique(y)

    for i, cls in enumerate(classes):
        # Let the current class be 1 and all other class be -1
        yBi = np.where(y == cls, 1, -1)
        for iter in range(max_iter):
            for j in range(n_samples):
                # Calculate the kernel value
                kernel_value = polykernel(sv, X[j].reshape(1,-1),degree)
                # Make prediction
                prediction = np.sign(np.sum(alpha[i]*yBi*kernel_value))
                # Update alpha if the prediction is wrong
                if prediction != yBi[j]:
                    alpha[i,j]+=1

    return alpha, sv, sv_labels, classes

# Implementation for kernelized Perceptron to predict on testing data
def predict(X, alpha, sv, sv_labels, classes, degree = 3):
    n_samples = X.shape[0]
    n_features = X.shape[1]
    pred=[]
    for i in range(n_samples):
        scoreList = []
        for j in range(len(classes)):
            kernel_value = polykernel(sv, X[i].reshape(1,-1),degree)
            score = np.sum(alpha[i]*kernel_value)
            scoreList.append(score)
        pred.append(classes[np.argmax(scoreList)])
    
    return np.array(pred)