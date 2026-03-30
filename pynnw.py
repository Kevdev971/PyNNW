import numpy as np

class NNW:
    def __init__(self, layer_sizes):
        # เช่น [2, 10, 10, 1]
        self.layers = layer_sizes
        
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes)-1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]))
            self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def sigm_di(self, x):
        return x*(1-x)

    def forward(self, x):
        self.activations = [x]  # เก็บทุก layer
        self.z_values = []

        a = x
        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = self.sigm(z)

            self.z_values.append(z)
            self.activations.append(a)
        return a

    def backward(self, x, y, lr):
        # --- output layer ---
        error = y - self.activations[-1]
        delta = error * self.sigm_di(self.activations[-1])

        # ไล่ย้อนทุก layer
        for i in reversed(range(len(self.weights))):
            a_prev = self.activations[i]

            w_old = self.weights[i].copy()

            # update weight
            self.weights[i] += np.dot(a_prev.T, delta) * lr
            self.biases[i] += np.sum(delta, axis=0, keepdims=True) * lr

            # คำนวณ delta ย้อน (ยกเว้น input layer)
            if i != 0:
                delta = np.dot(delta, w_old.T) * self.sigm_di(self.activations[i])

    def train(self, x, y, epochs, lr):
        for epoch in range(epochs):
            out = self.forward(x)
            self.backward(x, y, lr)

            if epoch % 4000 == 0:
                loss = np.mean((y - out)**2)
                print(f"epoch {epoch}, loss {loss}")

def array(x):
    return np.array(x)