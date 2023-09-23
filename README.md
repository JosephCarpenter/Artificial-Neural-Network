# Artificial Neural Network - Iris Database

## Overview
This program classifies 3 different types of flowers (Iris Setosa, Iris Versicolour, and Iris Virginica) using an Artificial Neural Network. Training data is obtained from Fisher’s Iris database (Fisher, 1936), which contains 50 instances of each type of Iris plant. 

By querying with 4 measurements as input (sepal length, sepal width, petal length, petal width), the program will return the best-fit Iris plant type.


The data is split in a 60:20:20 ratio for training, validation, and testing.
 The data trains until meeting a target MSE (0.1), or until a set tolerance for overfitting has been met. The target MSE, overfit values, learning rate, and initial bias values were chosen because they were shown to be efficient in testing. Likewise, the sigmoid function is used as the activation function because I found it to be the most accurate and efficient.

### Execution
To run, use the command:
`$ python3 main.py`
