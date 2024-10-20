import numpy as np
import mnist_reader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from SVM import svm, plot_svm, standardScale, find_best_C, svm_poly
from kernelizedPerceptron import polykernel, kernelizedPerceptron, predict

def main():
    X_train_full, y_train_full = mnist_reader.load_mnist('data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

    # Convert the label to binary [1, -1] where [1,3,5,7,9] are [-1] and [0,2,4,6,8] are [1]
    y_trainBi_full = np.where(y_train_full % 2 == 0, 1, -1)
    y_testBi = np.where(y_test % 2 == 0, 1, -1)

    # Split the dataset into trainning and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_trainBi_full, test_size=0.2, random_state=42)

    X_train, X_val, X_test = standardScale(X_train, X_val, X_test)

    C_values = [10**i for i in range(-4, 5)]
    trainAccuracy, valAccuracy, testAccuracy, sv = svm(X_train, y_train, X_val, y_val, X_test, y_testBi, C_values)
    # Print out the accuracies for each C value
    for i, C in enumerate(C_values):
        print(f"C = {C}:")
        print(f"Train Accuracy: {trainAccuracy[i]:.4f}")
        print(f"Validation Accuracy: {valAccuracy[i]:.4f}")
        print(f"Test Accuracy: {testAccuracy[i]:.4f}")

    # Plot the accuracies for each C value
    plot_svm(trainAccuracy, valAccuracy, testAccuracy, sv, C_values, kernel="Linear Kernel")

    # Output the best C value and the best accuracy
    best_C, best_accuracy = find_best_C(X_train, y_train, X_val, y_val, C_values)
    print(f"Best C: {best_C}")
    print(f"Best Accuracy: {best_accuracy:.4f}")

    # 2.1c Use polynomial kernel
    svm_poly(X_train, y_train, X_val, y_val, X_test, y_test, C_values)

    # 2.2 Implementation on kernelized Perceptron.
    # From previous observation, degree 3 has the best performance, although all three of them are similar.
    degree = 3
    max_iter = 5
    # train the kernelized perceptron
    alpha, sv, sv_labels, classes = kernelizedPerceptron(X_train, y_train, degree = degree, max_iter = max_iter)
    # make prediction with test and validation set.
    y_valPred = predict(X_val, alpha, sv, sv_labels, classes, degree=degree)
    y_testPred = predict(X_test, alpha, sv, sv_labels, classes, degree=degree)
    # calculate the accuracy for validation and test set
    val_acc = accuracy_score(y_val, y_valPred)
    test_acc = accuracy_score(y_test, y_testPred)

if __name__ == "__main__":

    main()