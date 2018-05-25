import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

#this adds a second hidden layer. Now weve got an input layer then a 6 neuron layer then a 4 neuron layer then output layer



class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],6) 
        self.weights2   = np.random.rand(6,4)
        self.weights3   = np.random.rand(4,1)                 
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights3 = np.dot(self.layer2.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T) * sigmoid_derivative(self.layer2)))
 #      d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.layer2), self.weights2.T) * sigmoid_derivative(self.layer1)))
 #       d_weights1 = np.dot(self.input.T, (np.dot((np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights3.T),self.weights2.T) * sigmoid_derivative(self.layer2))) * sigmoid_derivative(self.layer1))
        # update the weights with the derivative (slope) of the loss function
        d_weights1 = np.dot(self.input.T, np.dot(np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output),self.weights3.T) * sigmoid_derivative(self.layer2),self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += 0.5 * d_weights1
        self.weights2 += 0.5 * d_weights2
        self.weights3 += 0.5 * d_weights3



#original inputs were 3D, changed to 5D

if __name__ == "__main__":
    X = np.array([[0,0,1,0,1],
                  [0,1,1,1,1],
                  [1,0,1,0,0],
                  [1,1,1,1,1],
                  [0,0,0,0,0],
                  [1,1,0,0,0],
                  [0,1,1,0,0],
                  [0,1,0,1,0]])
    y = np.array([[0],[1],[0],[1],[0],[1],[1],[0]])
    nn = NeuralNetwork(X,y)

    Losshistory = []
    for i in range(1000):
        nn.feedforward()
        nn.backprop()
        Losshistory.append(sum((nn.y - nn.output)**2)/len((nn.y - nn.output)**2))
        
    print(nn.output)
    print((nn.y - nn.output)**2)
    print(len(Losshistory))
    
 #   plt.plot(list(range(500)),Losshistory)
#    plt.show()

def predict(x_newinput):
    # This function returns the predicted output
    # nn.weights1, nn.weights2 are obtained from the training - (ref. class NeuralNetwork)
    layer1 = sigmoid(np.dot(x_newinput, nn.weights1))
    layer2 = sigmoid(np.dot(layer1, nn.weights2))
    output = sigmoid(np.dot(layer2, nn.weights3))
    return output

print(predict([1,1,1,1,0]))

plt.plot(list(range(1000)),Losshistory)
plt.show()
