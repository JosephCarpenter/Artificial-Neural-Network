import csv
import numpy as np
import random


class ANN:
    
    # Load and preprocess iris data from dataset, initialize weights
    def __init__(self, data_file):
        
        self.bias = 1 # overall bias
        self.overfit_tolerance = 100
        self.target_mse = 0.1 # mean squared error to achieve
        self.learning_rate = 0.1
        
        with open(data_file, newline='') as csvfile:
            data = list(csv.reader(csvfile))
        
        # store all attributes and classes to be divided into subsets
        attributes = []
        classes = [] 

        # represent class identification as binary vector
        for datapoint in data:
            if len(datapoint) != 0:
                if datapoint[-1] == "Iris-setosa":   
                    classes.append([1, 0, 0])
                elif datapoint[-1] == "Iris-versicolor":
                    classes.append([0, 1, 0])
                elif datapoint[-1] == "Iris-virginica":
                    classes.append([0, 0, 1])
                attributes.append(list(map(float, datapoint[:-1])))

        # split data with 60:20:20 ratio into training, validation, and testing sets
        self.training_attributes = np.asarray(attributes[0:30] + attributes[50:80] + attributes[100:130]) / 10 # divide to normalize
        self.validation_attributes = np.asarray(attributes[30:40] + attributes[80:90] + attributes[140:150]) / 10
        self.testing_attributes = np.asarray(attributes[40:50] + attributes[90:100] + attributes[140:150]) / 10

        self.training_classes = np.asarray(classes[0:30] + classes[50:80] + classes[100:130])
        self.validation_classes = np.asarray(classes[30:40] + classes[80:90] + classes[130:140])
        self.testing_classes = np.asarray(classes[40:50] + classes[90:100] + classes[140:150])
       
        np.random.seed(524) # found that this seed results in efficient & accurate classification
        # 4 columns each corresponding to 4 Iris attributes
        self.weights_input = np.random.rand(4,4) # 4x4 weights matrix to use for hidden layer (4 rows for dot product compatibility)
        self.weights_hidden = np.random.rand(3,4) # 3x4 weights matrix to use for output layer (one row for each flower type)
        # create bias for each set of weights
        self.weights_input_biases = np.random.rand(4) 
        self.weights_hidden_biases = np.random.rand(3)  


    # train, validate, and test ANN using forward/backward propagation
    def train(self):
        
        training_mse = 1
        validation_mse = 1
        prev_training_mse = 1
        prev_validation_mse = 1
        overfit_counter = 0
        generation = 0

        gen_print = True if input("Print training and validation accuracy by generation? (y/n) ") == 'y' else False
        print("Training neural network...")
        
        # Train until achieving target MSE, or until exceeding the tolerance for overfitting
        while validation_mse > self.target_mse and overfit_counter < self.overfit_tolerance:
            
            # TRAINING
            generation += 1
            correct = 0
            squared_error = 0
            for datapoint, class_type in zip(self.training_attributes, self.training_classes):

                # Forward Propagation
                correct, squared_error, output_final, output_hidden = self.forward_propagation(datapoint, class_type, correct, squared_error)
                
                # Backward Propagation
                # calculate the error in final output with derivative of activation function
                error_output = self.activation(output_final, True) * (class_type - output_final)

                # calculate new weights between the hidden layer and the output layer
                self.weights_hidden += self.learning_rate * np.dot(error_output.reshape(3,1),output_hidden.reshape(4,1).T)
                self.weights_hidden_biases += self.learning_rate * self.bias * error_output

                # calculate error in the output of the hidden layer with derivative of activation function
                error_hidden = self.activation(output_hidden, True) * np.dot(self.weights_hidden.T, error_output)

                # calculate new weights between the input layer and the hidden
                # layer. weights_input is a 4 x 4 matrix
                self.weights_input += self.learning_rate * np.dot(error_hidden.reshape(4,1), datapoint.reshape(4,1).T)
                self.weights_input_biases += self.learning_rate * self.bias * error_hidden
            
            # accuracy and mse measurements
            training_mse = squared_error / 90
            if gen_print:
                print(f"Gen {generation}: ", end = "     ")
                print("Training Accuracy: {0:.2f}%".format(correct / 90 * 100), end="     ")
           
            prev_training_mse = training_mse

            # VALIDATION
            correct = 0
            squared_error = 0
            for datapoint, class_type in zip(self.validation_attributes, self.validation_classes):
                # only perform forward propagation to validate
                correct, squared_error = self.forward_propagation(datapoint, class_type, correct, squared_error)[:2]
            
            validation_mse = squared_error / 30
            if gen_print:
                print("Validation Accuracy: {0:.2f}%".format(correct / 30 * 100))
            prev_validation_mse = validation_mse

            # if validation mse increases while training mse decreases, continue approaching overfit tolerance
            if training_mse < prev_training_mse and validation_mse >= prev_validation_mse:
                overfit_counter += 1
                
        # TESTING
        correct = 0
        for datapoint, class_type in zip(self.testing_attributes, self.testing_classes):
            correct = self.forward_propagation(datapoint, class_type, correct, squared_error)[0]
        print("Testing accuracy: {0:.2f}%".format(correct / 30 * 100))


    # calculate output based on potential using sigmoid function
    def activation(self, p, prime=False):
        activation = 1/(1 + np.exp(-p))
        if prime: # return derivative
            return activation * (1 - activation)
        return activation


    # perform forward propagation for given datapoint
    def forward_propagation(self, datapoint, class_type, correct, squared_error=-1):

        # calculate potentials for hidden layer using input weights
        potentials_hidden = np.dot(self.weights_input, datapoint) + self.bias * self.weights_input_biases
        output_hidden = self.activation(potentials_hidden)

        # Calculates the potentials of the output layer and then calculates final output. 
        potentials_final = np.dot(self.weights_hidden, output_hidden) + self.bias * self.weights_hidden_biases
        output_final = self.activation(potentials_final)

        # add squared error
        if squared_error != -1:
            squared_error += np.sum((class_type - output_final)**2)
        
        # accumunate number of correct classes for accuracy calculation
        if class_type[np.argmax(output_final)] == 1:
            correct += 1

        return correct, squared_error, output_final, output_hidden
    
    # Allow user to query for flower type with given attributes
    def run(self):
        user_input = ""
        while user_input != "q":
            user_input = input("\nPlease enter 4 values (in centimeters), separated by a space: \n[sepal length] [sepal width] [petal length] [petal width].\nTo quit, enter q.\n")
            if user_input != "q":
                custom_input = np.asarray(list(map(float, user_input.split()))) / 10
                output_final = self.forward_propagation(custom_input, [0,0,0], 0)[2] # pass classtype [0,0,0] for unused correctness calculations
                class_types = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
                print(f"This is most likely to be an {class_types[np.argmax(output_final)]}.")


def main():
    ann = ANN("data.txt")
    ann.train()
    ann.run()


if __name__ == "__main__":
    main()