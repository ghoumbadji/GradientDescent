import numpy as np

# We'll work in a 2D dimension so for each point, we'll have (xi, yi(real) | yp(predicted))
# We'll use Mean Square Error (MSE) as the loss function as we have a regression problem
# By definition we have MSE = (1 / n) * [sum (yi - yp) for xi values]
# If we suppose that predicted value follows linear rule, we have yp = (m * x) + b
# So the developed formula of loss is MSE = (1 / n) * [sum (yi - ((m * x) + b)) for xi in values]
# For perfect model (model that doesn't exist), yi = yp so yi - yp = 0
# Then, the real goal of gradient descent is to ensure yp - yi ~= 0: we say we minimize error

class GDRegressor:
    def __init__(self, learning_rate=0.01, epochs=50):
        np.random.seed()
        self.m = np.random.randn()
        self.b = np.random.randn()
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("All arrays must have the same size.");
        for i in range(self.epochs):
            loss_slop_b = np.sum([-2 * (yi - (self.m*xi + self.b)) for xi,yi in zip(X,y)]) / len(X)
            loss_slop_m = np.sum([-2 * xi * (yi - (self.m*xi + self.b)) for xi,yi in zip(X,y)]) / len(X)
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
