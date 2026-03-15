'''softmax_layer.py
Constructs, trains, tests single layer neural network with softmax activation function.
YOUR NAMES HERE
CS 343: Neural Networks
Spring 2026
Project 2: Multilayer Perceptrons
''',
import numpy as np


class SoftmaxLayer:
    '''SoftmaxLayer is a class for single layer networks with softmax activation and cross-entropy loss
    in the output layer.
    '''
    def __init__(self, num_output_units):
        '''SoftmaxLayer constructor

        Parameters:
        -----------
        num_output_units: int. Num output units. Equal to # data classes.
        '''
        # Network weights
        self.wts = None
        # Bias
        self.b = None
        # Number of data classes C
        self.num_output_units = num_output_units

    def accuracy(self, y, y_pred):
        '''Computes the accuracy of classified samples. Proportion correct

        Parameters:
        -----------
        y: ndarray. int-coded true classes. shape=(Num samps,)
        y_pred: ndarray. int-coded predicted classes by the network. shape=(Num samps,)

        Returns:
        -----------
        float. accuracy in range [0, 1]
        '''
        # Convert to numpy arrays, only helpful if lists are passed
        y = np.asarray(y)
        y_pred = np.asarray(y_pred)
        # Check up
        if y.shape != y_pred.shape:
            raise ValueError(f"Shape Mismatch: y has shape {y.shape} and y_pred has shape {y_pred.shape}")
        # Propoertion correct
        return np.mean(y_pred == y)

    def net_in(self, features):
        '''Computes the net input (net weighted sum)

        Parameters:
        -----------
        features: ndarray. input data. shape=(num images (in mini-batch), num features)
        i.e. shape=(N, M)

        Note: shape of self.wts = (M, C), for C output neurons

        Returns:
        -----------
        net_input: ndarray. shape=(N, C)
        '''
        weighted_sum = np.dot(features, self.wts) # matrix-vector dot product, weighted sum per sample
        net_in = weighted_sum + self.b
        return net_in

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,C-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes (C) = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        N = y.shape[0] # number of samples
        C = num_classes 
        y_one_hot = np.zeros((N, C)) # initialize zeros
        y_one_hot[np.arange(N), y] = 1 # Places 1's at the correct location
        return y_one_hot
        

    def fit(self, features, y, n_epochs=100, lr=0.0001, mini_batch_sz=256, reg=0, r_seed=None, verbose=2):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features M)
        y: ndarray. int-coded class assignments of training samples. 0,...,numClasses-1
        n_epochs: int. Number of training epochs
        lr: float. Learning rate
        mini_batch_sz: int. Batch size per training iteration.
            i.e. Chunk this many data samples together to process with the model on each training
            iteration. Then we do gradient descent and update the wts. NOT the same thing as an epoch.
        reg: float. Regularization strength used when computing the loss and gradient.
        r_seed: None or int. Random seed for weight initialization.
        verbose: int. 0 means no print outs. Any value > 0 prints Current iteration number and training loss every
            100 iterations.

        Returns:
        -----------
        loss_history: Python list of floats. Recorded training loss on every mini-batch / training
            iteration.

        NOTE:
        Recall: training epoch is not the same thing as training iteration with mini-batch.
        If we have mini_batch_sz = 100 and N = 1000, then we have 10 iterations per epoch. Epoch
        still means entire pass through the training data "on average". Print this information out
        if verbose > 0.

        TODO:
        -----------
        1) Initialize the wts to small Gaussian numbers: mean 0, std 0.01, Wts shape=(num_feat M, num_classes C).
        2) Initialize the bias as usual with shape=(num_classes C,)
        2) Implement mini-batch support: On every iter draw from our input samples (with replacement)
        a batch of samples equal in size to `mini_batch_sz`. Also keep track of the associated labels.
        THEY MUST MATCH UP!!
            - Keep in mind that mini-batch wt updates are different than epochs. There is a parameter
              for E (num epochs), not number of iterations.
            - Handle this edge case: we do SGD and mini_batch_sz = 1. Add a singleton dimension
              so that the "N"/sample_size dimension is still defined.
        4) Our labels are int coded (0,1,2,3...) but this representation doesnt work well for piping
        signals to the C output neurons (C = num classes). Transform the mini-batch labels to one-hot
        coding from int coding (see function above to write this code).
        5) Compute the "net in".
        6) Compute the activation values for the output neurons (you can defer the actual function
        implementation of this for later).
        7) Compute the cross-entropy loss (again, you can defer the details for now)
        8) Do backprop:
            a) Compute the error gradient for the mini-batch sample,
            b) update weights using gradient descent.

        HINTS:
        -----------
        2) Work in indices, not data elements.
        '''
        X = np.asarray(features)
        y_int = np.asarray(y).astype(int)
        N, M = X.shape # Acquire shape of features
        C = self.num_output_units # number of classes
        # Print epoch/iteration info
        iters_per_epoch = int(np.ceil(N / mini_batch_sz))
        total_iters = n_epochs * iters_per_epoch
        if verbose > 0:
            print(f"N{N}, M={M}, C={C}")
            print(f"mini_batch_sz={mini_batch_sz} -> iters/epoch={iters_per_epoch}, total iters={total_iters}")
        # Initialize weights (to small Gaussian numbers) and bias
        rng = np.random.default_rng(r_seed)
        self.wts = rng.normal(loc=0.0, scale=0.01, size=(M, C))
        self.b = np.zeros(C, dtype=np.float64)
        loss_history = []
        # SGD training loop
        iter_count = 0
        for epoch in range(n_epochs):
            for _ in range(iters_per_epoch):
                iter_count += 1
                # Mini-batch sampling W/ replacement (work in indices)
                batch_idx = rng.integers(low=0, high=N, size=mini_batch_sz)
                X_mb = X[batch_idx]
                y_mb = y_int[batch_idx]
                # Edge case if mini_batch_sz = 1, keep X_mb 2D and y_mb 1D
                if X_mb.ndim == 1:
                    X_mb = X_mb[np.newaxis, :]
                if y_mb.ndim == 0:
                    y_mb = np.array([y_mb])
                # One-hot encode labels for mini-batch
                Y_mb =  self.one_hot(y_mb, C) # shape: (mini_batch_sz, C)
                # Net inputs
                Z = self.net_in(X_mb) # shape:(mini_batch_sz, C) 
                # Softmax (probabilities)
                S = self.activation(Z) # shape: (mini_batch_sz, C)
                # Loss
                L = self.loss(S, y_mb, reg=reg)
                loss_history.append(L)
                # Backdrop gradients and update params
                gradient_wts, gradient_b = self.gradient(X_mb, S, Y_mb, reg=reg)
                self.wts -= lr * gradient_wts
                self.b -= lr * gradient_b
                # Print verbosed
                if verbose > 0 and (iter_count % 100 == 0):
                    print(f"iter {iter_count:>6}/{total_iters} | epoch {epoch+1:>3}/{n_epochs} | loss {L:.4f}")
        return loss_history

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            Note: You can figure out the predicted class assignments from net_in (i.e. you dont
            need to apply the net activation function — it will not affect the most active neuron).
        '''
        # Checks
        if self.wts is None or self.b is None:
            raise ValueError("Model parameters are nro initialized, set up self.wts and self.b before predict()")
        # Compute logits (raw scores)
        net_in = self.net_in(features)
        # Predicted class = index of max score per sample
        y_pred = np.argmax(net_in, axis=1)
        return y_pred

    def activation(self, net_in):
        '''Applies the softmax activation function on the net_in.

        Parameters:
        -----------
        net_in: ndarray. net in. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)

        Returns:
        -----------
        f_z: ndarray. net_act transformed by softmax function. shape=(N, C)

        Tips:
        -----------
        - Remember the adjust-by-the-max trick (for each input samp) to prevent numeric overflow!
        This will make the max net_in value for a given input 0.
        - np.sum and np.max have a keepdims optional parameter that might be useful for avoiding
        going from shape=(X, Y) -> (X,). keepdims ensures the result has shape (X, 1).
        '''
        # get numerical stability by subtracting row-wise max
        z = net_in - np.max(net_in, axis=1, keepdims=True)
        # Exponentiate
        z_exponent = np.exp(z)
        # Normalize row-wise
        f_z = z_exponent / np.sum(z_exponent, axis=1, keepdims=True)
        return f_z
        
    def loss(self, net_act, y, reg=0):
        '''Computes the cross-entropy loss

        Parameters:
        -----------
        net_act: ndarray. softmax net activation. shape=(mini-batch size, num output neurons)
        i.e. shape=(N, C)
        y: ndarray. correct class values, int-coded. shape=(mini-batch size,)
        reg: float. Regularization strength

        Returns:
        -----------
        loss: float. Regularized (!!!!) average loss over the mini batch

        Tips:
        -----------
        - Remember that the loss is the negative of the average softmax activation values of neurons
        coding the correct classes only.
        - It is handy to use arange indexing to select only the net_act values coded by the correct
          output neurons.
        - NO FOR LOOPS!
        - Remember to add on the regularization term, which has a 1/2 in front of it.
        '''
        # Get number of samples
        N = net_act.shape[0]
        # Get probability of the correct classes
        p_correct = net_act[np.arange(N), y]
        # Cross-entropy loss (avg. over batch)
        data_loss = -np.mean(np.log(p_correct + 1e-12)) # +eps prevents log(0)
        # L2 regularization on weights only
        regularization_loss = (reg / 2) * np.sum(self.wts ** 2)
        # Total loss
        total_loss = data_loss + regularization_loss
        return total_loss

    def gradient(self, features, net_act, y, reg=0):
        '''Computes the gradient of the softmax version of the net

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        net_act: ndarray. net outputs. shape=(mini-batch-size, C)
            In the softmax network, net_act for each input has the interpretation that
            it is a probability that the input belongs to each of the C output classes.
        y: ndarray. one-hot coded class labels. shape=(mini-batch-size, Num output neurons)
        reg: float. regularization strength.

        Returns:
        -----------
        grad_wts: ndarray. Weight gradient. shape=(Num features, C)
        grad_b: ndarray. Bias gradient. shape=(C,)

        NOTE:
        - Gradient is the same as ADALINE, except we average over mini-batch in both wts and bias.
        - NO FOR LOOPS!
        - Don't forget regularization!!!! (Weights only, not for bias)
        '''
        # Checks
        if self.wts is None or self.b is None:
            raise ValueError("Model parameters are nro initialized, set up self.wts and self.b before predict()")
        # mini-batch size
        N = features.shape[0]
        # dL / dZ = S - Y
        dZ = net_act - y
        # dL / dW = (1/N)X^T (S - Y) + reg * W
        gradient_wts = (features.T @ dZ) / N
        gradient_wts += reg * self.wts
        # dL / db = (1/N) * sum over samples of (S - Y)
        gradient_bias = np.sum(dZ, axis=0) / N # Shape: (C, )
        return gradient_wts, gradient_bias

    def test_loss(self, wts, b, features, labels):
        ''' Tester method for net_in and loss
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in shape={net_in.shape}, min={net_in.min()}, max={net_in.max()}')
        print('Should be\nnet in shape=(15, 10), min=0.584664799299611, max=1.411396670099296\n')

        net_act = self.activation(net_in)
        print(f'net act shape={net_act.shape}, min={net_act.min()}, max={net_act.max()}')
        print('Should be\nnet act shape=(15, 10), min=0.0665134672262976, max=0.1439281981621258\n')
        return self.loss(net_act, labels, 0), self.loss(net_act, labels, 0.5)

    def test_gradient(self, wts, b, features, labels, num_unique_classes, reg=0):
        ''' Tester method for gradient
        '''
        self.wts = wts
        self.b = b

        net_in = self.net_in(features)
        print(f'net in: {net_in.shape}, {net_in.min()}, {net_in.max()}')
        print(f'net in 1st few values of 1st input are:\n{net_in[0, :5]}\nand should be')
        print('[0.798 1.095 0.969 0.9   0.958]')

        net_act = self.activation(net_in)
        print(f'net act 1st few values of 1st input are:\n{net_act[0, :5]}\nand should be')
        print('[0.078 0.105 0.092 0.086 0.091]')

        labels_one_hot = self.one_hot(labels, num_unique_classes)
        print(f'y one hot: {labels_one_hot.shape}, sum is {np.sum(labels_one_hot)}.')
        print('You should know what the sum should be :)')

        return self.gradient(features, net_act, labels_one_hot, reg=reg)
