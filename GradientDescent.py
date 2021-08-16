import numpy as np
import random as rd

#Suppose
# yi = real correspondance of xi
# yp = predict value of a xi through the model
# f = loss function (we'll use mean square error (mse) here)
# By definition we have f = (1 / n) * [sum (yi - yp) for xi values]
# If we suppose that predict value follows linear rule, we have yp = (m * x) + b
# So the developed formula of loss is f = (1 / n) * [sum (yi - (m * x)) for xi in values]
# For perfect model (model that doesn't exist), we would have yi = yp so yi - yp would be zero
# Then, the real goal is to be closed to zero: we say we minimize error
# This is the goal of gradient descent
# Let's get started

class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=50):
        rd.seed()
        self.m = rd.randint(1,9)
        self.b = rd.randint(1,9)
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("All arrays must have the same size.");
        for i in range(self.epochs):
            #partial derivative of f with respect to b
            loss_slop_b = np.sum([-2 * (yi - (self.m*xi + self.b)) for xi,yi in zip(X,y)]) / len(X)
            #partial derivative of f with respect to b
            loss_slop_m = np.sum([-2 * xi * (yi - (self.m*xi + self.b)) for xi,yi in zip(X,y)]) / len(X)
            #minimalize error by minimizng b and m
            self.b -= (self.lr * loss_slop_b)
            self.m -= (self.lr * loss_slop_m)

    def predict(self, x):
        return self.m * x + self.b

#if __name__ == "__main__":
#    X = np.array([0, 1, 2, 3, 4, 5], dtype=float)
#    y = np.array([0, 2, 4, 6, 8, 10], dtype=float)
#    model = GDRegressor(0.01, 100)
#    model.fit(X, y)
#    print(model.predict(6))
