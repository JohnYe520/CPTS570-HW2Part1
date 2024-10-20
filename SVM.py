import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# preprocessing data since the max_iter is set to 1000
def standardScale(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test

# 2.1a Linear Kernel
def svm(X_train, y_train, X_val, y_val, X_test, y_test, C_values):
    trainAccuracy = []
    valAccuracy = []
    testAccuracy = []
    sv = []
    for C in C_values:
        # Initialize the classifier
        clf = SVC(C=C, kernel='linear',max_iter=1000)

        # This is the default value for full iteration, however, the code runs for over
        # 30 minutes without stopping, so I have to set the max_iter to 1000
        # clf = SVC(C=C, kernel='linear')

        clf.fit(X_train, y_train)

        # predict the training data 
        y_pred_train = clf.predict(X_train)
        # calculate the accuracy of the training data
        trainAccuracy.append(accuracy_score(y_train, y_pred_train))

        # predict the validation data
        y_pred_val = clf.predict(X_val)
        # calculate the accuracy of the validation data
        valAccuracy.append(accuracy_score(y_val, y_pred_val))

        # predict the test data
        y_pred_test = clf.predict(X_test)
        # calculate the accuracy of the test data
        testAccuracy.append(accuracy_score(y_test, y_pred_test))

        # get the support vectors
        sv.append(clf.support_vectors_)

    return trainAccuracy, valAccuracy, testAccuracy, sv

# 2.1a Plot the accuracies for each C value
def plot_svm(trainAccuracy, valAccuracy, testAccuracy, sv, C_values, kernel):
    plt.plot(C_values, trainAccuracy, label='Train Accuracy')
    plt.plot(C_values, valAccuracy, label='Validation Accuracy')
    plt.plot(C_values, testAccuracy, label='Test Accuracy')
    plt.xlabel('C Values')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs C Values')
    plt.legend()
    plt.show()
    
    plt.plot(C_values, [len(s) for s in sv], label='Support Vectors')
    plt.xlabel('C Values')
    plt.ylabel('Support Vectors')
    plt.title('Support Vectors vs C Values')
    plt.legend()
    plt.show()

# 2.1b Find the best C value
def find_best_C(X_train, y_train, X_val, y_val, C_values):
    best_C = None
    best_accuracy = 0
    for C in C_values:
        clf = SVC(C=C, kernel='linear',max_iter=1000)
        clf.fit(X_train, y_train)
        accuracy = accuracy_score(y_val, clf.predict(X_val))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = C
    return best_C, best_accuracy

# 2.1c Use polynomial kernel
def svm_poly(X_train, y_train, X_val, y_val, X_test, y_test, C_values):
    for degree in [2,3,4]:
        print(f"Degree {degree}:")
        trainAccuracy = []
        valAccuracy = []
        testAccuracy = []
        sv = []
        for C in C_values:
            clf = SVC(C=C, kernel='poly', degree=degree, max_iter=1000)
            # Same as above, set the max_iter to 1000, the below code is for full iteration
            # clf = SVC(C=C, kernel='poly', degree=degree)

            clf.fit(X_train, y_train)

            # predict the training data 
            y_pred_train = clf.predict(X_train)
            # calculate the accuracy of the training data
            trainAccuracy.append(accuracy_score(y_train, y_pred_train))

            # predict the validation data
            y_pred_val = clf.predict(X_val)
            # calculate the accuracy of the validation data
            valAccuracy.append(accuracy_score(y_val, y_pred_val))

            # predict the test data
            y_pred_test = clf.predict(X_test)
            # calculate the accuracy of the test data
            testAccuracy.append(accuracy_score(y_test, y_pred_test))

            # get the support vectors
            sv.append(clf.support_vectors_)

        plot_svm(trainAccuracy, valAccuracy, testAccuracy, sv, C_values, kernel=f"Polynomial Kernel Degree {degree}")
