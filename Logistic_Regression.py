import numpy as np

class LogisticRegression:
    def __init__(self, X, Y, W, b, m, learning_rate):
        self.X = X
        self.Y = Y
        self.W = W
        self.b = b
        self.m = m
        self.learning_rate = learning_rate
    def sigmoid(self):
        self.z = np.dot(self.W.T, self.X) + self.b
        self.s = 1 / (1 + np.exp(-self.z))
        return self.s

    def cost(self):
        self.J = - np.sum(np.dot(self.Y, np.log(self.s)) + np.dot((1 - self.Y), np.log(1-self.s))) / self.m 
        return self.J
        
    def dW(self):
        self.A = self.sigmoid()
        self.var_dW = (1 / self.m) * np.dot(self.X, (self.A - self.Y).T)
        return self.var_dW
    def dB(self):
        self.A = self.sigmoid()
        self.var_dB = (1 / self.m) * np.sum(self.A - self.Y)
        return self.var_dB
    def update(self):
        self.W = self.W - self.learning_rate * self.var_dW
        self.b = self.b - self.learning_rate * self.var_dB
        return{
          "W":self.W,
          "b":self.b
        }
        

X = np.array([[1, 2, 4], [2, 3, 5], [3, 4, 6], [4, 5, 6]])
Y = np.array([1, 0, 1])
W = np.array([0, 0, 0, 0])
b = 0
m = 3

learning_rate = 0.009
epoch = 200000
itr = 0

nn = LogisticRegression(X, Y, W, b, m, learning_rate)

A = nn.sigmoid()
var_cost = nn.cost()
print(var_cost)

while(itr <= epoch):
    nn.dW()
    nn.dB()
    params = nn.update()
    A = nn.sigmoid()
    var_cost = nn.cost()
    print(var_cost)
    itr += 1

print(params["W"])
print(params["b"])