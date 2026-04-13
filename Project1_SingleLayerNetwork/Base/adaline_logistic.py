'''adaline_logistic.py
Duilio Lucio, Vivian Hu
CS343: Neural Networks
Project 1: Single Layer Networks
'''

import numpy as np
from adaline import Adaline

class AdalineLogistic(Adaline):
    
    def activation(self, net_in):
        """
        Sigmoid activation for logistic regression
        """
        return 1.0 / (1.0 + np.exp(-net_in))
    
    def loss(self, y, net_act):
        """  
        Cross-entropy loss
        """
        eps = 1e-15 # prevents log(0)
        net_act = np.clip(net_act, eps, 1 - eps)
        return -np.mean(y * np.log(net_act) + (1 - y) * np.log(1 - net_act))
        
    def predict(self, features):
        """  
        Class Prediction: threshold at 0.5
        """
        net_in = self.net_input(features)
        net_act = self.activation(net_in)
        return np.where(net_act >= 0.5, 1, 0)