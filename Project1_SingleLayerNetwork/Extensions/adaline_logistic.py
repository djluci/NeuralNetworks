'''
Vivian Hu and Duilio Lucio 
adaline_logistic.py
Project 1 Extension 
'''

import numpy as np
from adaline import Adaline


class AdalineLogistic(Adaline):
    """Single-layer logistic regression (ADALINE-style API)."""

    @staticmethod
    def _sigmoid(z):
        #clip prevents overflow in exp() which is important if z becomes large
        #the output is always between 0 and 1
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def activation(self, net_in):
        return self._sigmoid(net_in)

    def loss(self, y, net_act):
        eps = 1e-15
        net_act = np.clip(net_act, eps, 1 - eps)
        return -np.mean(y * np.log(net_act) + (1 - y) * np.log(1 - net_act))

    def predict(self, features):
        net_in = self.net_input(features) # computes z
        net_act = self.activation(net_in) #computes p
        return (net_act >= 0.5).astype(int)

    def fit(self, features, y, n_epochs=1000, lr=0.001, r_seed=None):
        N, M = features.shape
        self.loss_history = []
        self.accuracy_history = []

        rng = np.random.default_rng(r_seed)
        self.wts = rng.normal(loc=0.0, scale=0.01, size=M)
        self.b = 0.0

        y = y.astype(float)

        for _ in range(n_epochs):
            net_in = self.net_input(features)
            net_act = self.activation(net_in)

            delta = (net_act - y) / N  # (N,)

            grad_wts = features.T @ delta  # (M,)
            grad_b = float(np.sum(delta))

            self.wts -= lr * grad_wts
            self.b -= lr * grad_b

            self.loss_history.append(self.loss(y, net_act))
            self.accuracy_history.append(self.accuracy(y, self.predict(features)))

        return self.loss_history, self.accuracy_history


class AdalineGatedLogistic(AdalineLogistic):
    """Single-layer logistic regression + learnable per-feature gates.

    gates = sigmoid(g_raw) in (0,1)
    net_in = (features * gates) @ wts + b
    """

    def __init__(self, gate_init=0.0):
        super().__init__()
        self.g_raw = None
        self.gate_init = gate_init

    def get_gates(self):
        if self.g_raw is None:
            return None
        return self._sigmoid(self.g_raw)

    def net_input(self, features):
        gates = self.get_gates()
        if gates is None:
            return super().net_input(features)
        return (features * gates) @ self.wts + self.b

    def fit(self, features, y, n_epochs=1000, lr=0.001, r_seed=None, l1_gate=0.0):
        N, M = features.shape
        self.loss_history = []
        self.accuracy_history = []

        rng = np.random.default_rng(r_seed)
        self.wts = rng.normal(loc=0.0, scale=0.01, size=M)
        self.b = 0.0
        self.g_raw = np.full(M, float(self.gate_init), dtype=float)

        y = y.astype(float)

        for _ in range(n_epochs):
            gates = self._sigmoid(self.g_raw)  # (M,)
            Xg = features * gates              # (N, M)

            net_in = Xg @ self.wts + self.b
            net_act = self.activation(net_in)

            delta = (net_act - y) / N

            grad_wts = Xg.T @ delta
            grad_b = float(np.sum(delta))

            grad_gate = self.wts * (features.T @ delta)          # (M,)
            grad_g_raw = grad_gate * gates * (1.0 - gates)       # (M,)

            if l1_gate != 0.0:
                grad_g_raw += l1_gate * gates * (1.0 - gates)

            self.wts -= lr * grad_wts
            self.b -= lr * grad_b
            self.g_raw -= lr * grad_g_raw

            self.loss_history.append(self.loss(y, net_act))
            self.accuracy_history.append(self.accuracy(y, self.predict(features)))

        return self.loss_history, self.accuracy_history