'''adaline.py
Duilio Lucio, Vivian Hu
CS343: Neural Networks
Project 1: Single Layer Networks
ADALINE (ADaptive LInear NEuron) neural network for classification and regression
'''
import numpy as np


class Adaline():
    ''' Single-layer neural network

    Network weights are organized [wt1, wt2, wt3, ..., wtM] for a net with M input neurons.
    Bias is stored separately from wts.
    '''
    def __init__(self):
        '''ADALINE Constructor
        '''
        # Network weights: wt for input neuron 1 is at self.wts[0], wt for input neuron 2 is at self.wts[1], etc
        self.wts = None
        # Bias: will be a scalar
        self.b = None
        # Record of training loss. Will be a list. Value at index i corresponds to loss on epoch i.
        self.loss_history = None
        # Record of training accuracy. Will be a list. Value at index i corresponds to acc. on epoch i.
        self.accuracy_history = None

    def get_wts(self):
        ''' Returns a copy of the network weight array'''
        if self.wts is None:
            return None
        return self.wts.copy()

    def get_bias(self):
        ''' Returns a copy of the bias'''
        return self.b

    def net_input(self, features):
        ''' Computes the net_input (weighted sum of input features,  wts, bias)

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The net_input. Shape = [Num samples,]
        '''
        weighted_sum = np.dot(features, self.wts) # matrix-vector dot product, weighted sum per sample
        net_input = weighted_sum + self.b 
        return net_input
        

    def activation(self, net_in):
        '''Applies the activation function to the net input and returns the output neuron's activation.
        It is simply the identify function for vanilla ADALINE: f(x) = x

        Parameters:
        ----------
        net_in: ndarray. Shape = [Num samples N,]

        Returns:
        ----------
        net_act. ndarray. Shape = [Num samples N,]
        '''
        net_act = net_in
        return net_act

    def predict(self, features):
        '''Predicts the class of each test input sample

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        The predicted classes (-1 or +1) for each input feature vector. Shape = [Num samples N,]

        NOTE: Remember to apply the activation function!
        '''
        net_input = self.net_input(features)
        activation = self.activation(net_input)
        predictions = np.where(activation >= 0, 1, -1)
        return predictions

    def accuracy(self, y, y_pred):
        ''' Computes accuracy (proportion correct) (across a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch  (coded as -1 or +1).
        y_pred: ndarray. Shape = [Num samples N,]
            Predicted classes corresponding to each input sample (coded as -1 or +1).

        Returns:
        ----------
        float. The accuracy for each input sample in the epoch. ndarray.
            Expressed as proportions in [0.0, 1.0]
        '''
        accuracy = np.mean(y == y_pred)
        return accuracy

    def loss(self, y, net_act):
        ''' Computes the Sum of Squared Error (SSE) loss (over a single training epoch)

        Parameters:
        ----------
        y: ndarray. Shape = [Num samples N,]
            True classes corresponding to each input sample in a training epoch (coded as -1 or +1).
        net_act: ndarray. Shape = [Num samples N,]
            Output neuron's activation value (after activation function is applied)

        Returns:
        ----------
        float. The SSE loss (across a single training epoch).
        '''
        error = y - net_act
        return 0.5 * np.sum(error**2)
        

    def gradient(self, errors, features):
        ''' Computes the error gradient of the loss function (for a single epoch).
        Used for backpropogation.

        Parameters:
        ----------
        errors: ndarray. Shape = [Num samples N,]
            Difference between class and output neuron's activation value
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.

        Returns:
        ----------
        grad_bias: float.
            Gradient with respect to the bias term
        grad_wts: ndarray. shape=(Num features N,).
            Gradient with respect to the neuron weights in the input feature layer
        '''
        grad_bias = -np.sum(errors)
        grad_wts = -np.sum(errors[:, np.newaxis] * features, axis = 0)
        return grad_bias, grad_wts

    def fit(self, features, y, n_epochs=1000, lr=0.001, r_seed=None):
        '''Trains the network on the input features for self.n_epochs number of epochs

        Parameters:
        ----------
        features: ndarray. Shape = [Num samples N, Num features M]
            Collection of input vectors.
        y: ndarray. Shape = [Num samples N,]
            Classes corresponding to each input sample (coded -1 or +1).
        n_epochs: int.
            Number of epochs to use for training the network
        lr: float.
            Learning rate used in weight updates during training
        r_seed: None or int.
            Random seed used for controlling the reproducability of the wts

        Returns:
        ----------
        self.loss_history: Python list of network loss values for each epoch of training.
            Each loss value is the loss over a training epoch.
        self.acc_history: Python list of network accuracy values for each epoch of training
            Each accuracy value is the accuracy over a training epoch.

        TODO:
        1. Initialize the weights according to a Gaussian distribution centered at 0 with standard deviation of 0.01
        using the recommended method of generating random numbers from lecture.
        2. Initialize the bias according to the recommended method from lecture.
        3. Write the main training loop where you:
            - Pass the inputs in each training epoch through the net.
            - Compute the error, loss, and accuracy (across the entire epoch).
            - Do backprop to update the weights and bias.
        '''
        N, M = features.shape
        self.loss_history = []
        self.accuracy_history = []
        rng = np.random.default_rng(r_seed)
        # Initialize weights and bias
        self.wts = rng.normal(loc=0.0, scale=0.01, size=M)
        self.b = 0.0
        for _ in range(n_epochs):
            # Forward pass
            net_in = self.net_input(features)
            net_act = self.activation(net_in)
            # Error 
            errors = y - net_act # (N,)
            # Loss (SSE) over epoch
            epoch_loss = self.loss(y, net_act) # scalar
            self.loss_history.append(epoch_loss)
            # Accuracy over epoch
            y_pred = self.predict(features) # (N,)
            epoch_accuracy = self.accuracy(y, y_pred) # Scalar
            self.accuracy_history.append(epoch_accuracy)
            # Gradients 
            gradient_bias, gradient_wts = self.gradient(errors, features)
            # Gradient descent update
            self.b -= lr * gradient_bias
            self.wts -= lr *gradient_wts
            
        return self.loss_history, self.accuracy_history
            
class Perceptron(Adaline):
    """
    Perceptron classifier: same structure as Adaline, but activation is a step/sign function
    """
    def activation(self, net_in):
        """
        Step Activation:
        +1 if net_in >0
        -1 otherwise
        """
        return np.where(net_in >= 0, 1, -1)