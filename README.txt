Homework 2
John Ye
11581172
---------------------------------------------------
Programming Language: Python 3.12.6
---------------------------------------------------
Required packages:
- numpy
- pandas
- matplotlib
- sklearn
---------------------------------------------------
Usage:
To run this script:
1. Download the Fashion MNIST data from https://github.com/zalandoresearch/fashion-mnist
2. Place the Fashion MNIST data in the following directory:
    /data/fashion
    Make sure the /data are in the same folder as the code.
3. Run the script with the following command:
   python main.py
---------------------------------------------------
Project Structure for Decision Tree Part:
- main.py                   : Main Python script that read from the dataset
- SVM.py                    : Implementation for Support Vector Machine, plot the result, finding the best C-value, and polynomial kernel
- kernelizedPerceptron.py   : Implementation for Kernelized Perceprton
- mnist_reader.py           : Read from the dataset
                          (Sample code from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py)
- README.txt                : Project documentation
- 11581172-Ye.pdf           : Homework solution for analytical part and empirical analysis question for Part-II