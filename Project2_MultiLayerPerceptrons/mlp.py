'''mlp.py
Constructs, trains, tests 2 layer multilayer layer perceptron networks
Duilio Lucio, Vivian Hu
CS 343: Neural Networks
Spring 2026
Project 2: Multilayer Perceptrons
'''
import numpy as np


class MLP:
    '''MLP is a class for multilayer perceptron network.

    The structure of our MLP will be:

    Input layer (X units) ->
    Hidden layer (Y units) with Rectified Linear activation (ReLu) ->
    Output layer (Z units) with softmax activation

    Due to the softmax, activation of output neuron i represents the probability that the current input sample belongs
    to class i.
    '''
    def __init__(self, num_input_units, num_hidden_units, num_output_units, r_seed=None):
        '''Constructor to build the model structure and intialize the weights. There are 3 layers:
        input layer, hidden layer, and output layer. Since the input layer represents each input
        sample, we don't learn weights for it.

        Parameters:
        -----------
        num_input_units: int. Num input features
        num_hidden_units: int. Num hidden units
        num_output_units: int. Num output units. Equal to # data classes.
        r_seed: int or None. Random seed to control how weights are initialized.
        '''
        self.num_input_units = num_input_units
        self.num_hidden_units = num_hidden_units
        self.num_output_units = num_output_units

        self.initialize_wts(num_input_units, num_hidden_units, num_output_units, r_seed=r_seed)

    def get_y_wts(self):
        '''Returns a copy of the hidden layer wts'''
        return self.y_wts.copy()

    def initialize_wts(self, M, H, C, std=0.1, r_seed=None):
        '''Initialize the hidden and output layer weights and bias term

        Parameters:
        -----------
        M: int. Num input features
        H: int. Num hidden units
        C: int. Num output units. Equal to # data classes.
        std: float. Standard deviation of the normal distribution of weights
        r_seed: None or int. Random seed for weight initialization.

        Returns:
        -----------
        No return

        TODO:
        - Initialize self.y_wts and self.z_wts with the appropriate size according to the normal distribution with
        standard deviation `std` and mean of 0. For consistency with the test code, initialize the parameters in that
        order.
          - The wt shapes should be be equal to (#prev layer units, #associated layer units)
            for example: self.y_wts has shape (M, H)
        - Initialize the biases (self.y_b and self.z_b) using the usual strategy.
          - For bias shapes, they should equal the number of units in the associated layer.
            for example: self.y_b has shape (H,)
        '''
        rng = np.random.default_rng(r_seed)
        
        # Intialize wts w/ shape (input_dim, H)
        self.y_wts = rng.normal(loc=0.0, scale=std, size=(M, H))
        # Initialize wts w/ shape (H, C)
        self.z_wts = rng.normal(loc=0.0, scale=std, size=(H, C))
        
        # Initialize bias w/ shape (H,)
        self.y_b = np.zeros(H)
        # Initialize bias w/ shape (C,)
        self.z_b = np.zeros(C)
        
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

    def one_hot(self, y, num_classes):
        '''One-hot codes the output classes for a mini-batch

        Parameters:
        -----------
        y: ndarray. int-coded class assignments of training mini-batch. 0,...,numClasses-1
        num_classes: int. Number of unique output classes total

        Returns:
        -----------
        y_one_hot: One-hot coded class assignments.
            e.g. if y = [0, 2, 1] and num_classes = 4 we have:
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]]
        '''
        # One hot encoded used for correct classes
        N = y.shape[0] # number of samples
        C = num_classes 
        y_one_hot = np.zeros((N, C)) # initialize zeros
        y_one_hot[np.arange(N), y] = 1 # Places 1's at the correct location
        return y_one_hot

    def predict(self, features):
        '''Predicts the int-coded class value for network inputs ('features').

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. shape=(mini-batch size, num features)

        Returns:
        -----------
        y_pred: ndarray. shape=(mini-batch size,).
            This is the int-coded predicted class values for the inputs passed in.
            NOTE: You can figure out the predicted class assignments without applying the
            softmax net activation function — it will not affect the most active neuron.
        '''
        # compute forward pass to get output layer scores
        # output_layer output is usually (batch_size, C)
        # compute forward pass without y for predictions
        _, _, _, z_net_act, _ = self.forward(features, y=None)
        y_pred = np.argmax(z_net_act, axis=1)
        return y_pred

    def forward(self, features, y, reg=0):
        '''Performs a forward pass of the net (input -> hidden -> output).
        This should start with the features and progate the activity to the output layer, ending with the cross-entropy
        loss computation.

        Don't forget to add the regularization to the loss!

        NOTE: Implement all forward computations within this function
        (don't divide up into separate functions for net_in, net_act). Doing this all in one method is not good design,
        but as you will discover, having the forward computations (y_net_in, y_net_act, etc) easily accessible in one
        place makes the backward pass a lot easier to track during implementation. In future projects, we will rely on
        better OO design.

        NOTE: Loops of any kind are NOT ALLOWED in this method!

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size N, Num features M)
        y: ndarray. int coded class labels. shape=(mini-batch-size N,)
        reg: float. regularization strength.

        Returns:
        -----------
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        loss: float. REGULARIZED loss derived from output layer, averaged over all input samples

        NOTE:
        - To regularize loss for multiple layers, you add the usual regularization to the loss
          from each set of weights (i.e. 2 in this case).
        '''
        # Computes hidden layer output
        y_net_in = features @ self.y_wts + self.y_b # Shape: (N, H)
        # Apply ReLU
        y_net_act = np.maximum(0, y_net_in) # Shape: (N, H)
        # Compute output layer net Input
        z_net_in = y_net_act @ self.z_wts + self.z_b 
        
        # Softmax Act
        # Subtract max from each row to improve numerical stability
        z_net_stable = z_net_in - np.max(z_net_in, axis=1, keepdims=True)
        exp_scores = np.exp(z_net_stable)
        z_net_act = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # Shape: (N, C)
        
        # Cross-Entropy Loss
        # Convert true labels to one-hot
        # Only compute loss if y is not None
        if y is not None:
            y_one_hot = self.one_hot(y, self.z_wts.shape[1])
            correct_logprobs = -np.log(z_net_act[np.arange(len(y)), y])
            data_loss = np.mean(correct_logprobs)
            reg_loss = 0.5 * reg * (np.sum(self.y_wts ** 2) + np.sum(self.z_wts ** 2))
            loss = data_loss + reg_loss
        else:
            loss = None
        return y_net_in, y_net_act, z_net_in, z_net_act, loss

    def backward(self, features, y, y_net_in, y_net_act, z_net_in, z_net_act, reg=0):
        '''Performs a backward pass (output -> hidden -> input) during training to update the weights. This function
        implements the backpropogation algorithm.

        This should start with the loss and progate the activity backwards through the net to the input-hidden weights.

        I added dz_net_act for you to start with, which is your cross-entropy loss gradient.
        Next, tackle dz_net_in, dz_wts and so on.

        I suggest numbering your forward flow equations and process each for relevant gradients in reverse order until
        you hit the first set of weights.

        Don't forget to backpropogate the regularization to the weights! (I suggest worrying about this last)

        Parameters:
        -----------
        features: ndarray. net inputs. shape=(mini-batch-size, Num features)
        y: ndarray. int coded class labels. shape=(mini-batch-size,)
        y_net_in: ndarray. shape=(N, H). hidden layer "net in"
        y_net_act: ndarray. shape=(N, H). hidden layer activation
        z_net_in: ndarray. shape=(N, C). output layer "net in"
        z_net_act: ndarray. shape=(N, C). output layer activation
        reg: float. regularization strength.

        Returns:
        -----------
        dy_wts, dy_b, dz_wts, dz_b: The following backwards gradients
        (1) hidden wts, (2) hidden bias, (3) output weights, (4) output bias
        Shapes should match the respective wt/bias instance vars.

        NOTE:
        - Regularize each layer's weights like usual.
        '''
        N = features.shape[0] # to get N dimensions
        
        # Step 1. Computing z gradient (output gradient)
        dz_net_in = z_net_act.copy()
        dz_net_in[np.arange(N), y] -= 1
        dz_net_in /= N
        
        # Step 2. Gradients for output weights and bias
        dz_wts = y_net_act.T @ dz_net_in
        dz_b = np.sum(dz_net_in, axis=0)
        dz_wts += reg * self.z_wts # Adding in the regularization strength
        
        # Step 3. backdrop to hidden layer
        dy_net_in = dz_net_in @ self.z_wts.T
        dy_net_in[y_net_in <= 0] = 0 
        # Gradients for hidden wts and bias
        dy_wts = features.T @ dy_net_in
        dy_b = np.sum(dy_net_in, axis=0)
        dy_wts += reg * self.y_wts  # Add regularization again
        
        return dy_wts, dy_b, dz_wts, dz_b

    def fit(self, features, y, x_validation, y_validation, n_epochs=500, lr=0.0001, mini_batch_sz=256, reg=0,
            r_seed=None, verbose=2, print_every=100):
        '''Trains the network to data in `features` belonging to the int-coded classes `y`.
        Implements stochastic mini-batch gradient descent. The flow of this method should follow the one that you wrote
        in `SoftmaxLayer`.

        Changes from `fit` in `SoftmaxLayer`:
        -------------------------------------
        1. Starting with this MLP, we are moving weight and bias initialization to the class constructor (see above).
        So you do NOT need to initialize the wts and bias in this method.
        2. Remember to update weights and biases for ALL layers!
        3. Record accuracy on the validation set (`x_validation`, `y_validation`) after each epoch training.
        4. Record accuracy on training set after each epoch training.

        (see note below for more details about changes 3 and 4)

        Parameters:
        -----------
        features: ndarray. shape=(Num samples N, num features).
            Features over N inputs.
        y: ndarray.
            int-coded class assignments of training samples. 0,...,numClasses-1
        x_validation: ndarray. shape=(Num samples in validation set, num features).
            This is used for computing/printing the accuracy on the validation set at the end of each epoch.
        y_validation: ndarray.
            int-coded class assignments of validation samples. 0,...,numClasses-1
        n_epochs: int.
            Number of training epochs
        lr: float.
            Learning rate
        mini_batch_sz: int.
            Batch size per epoch. i.e. How many samples we draw from features to pass through the model per training epoch
            before we do gradient descent and update the wts.
        reg: float.
            Regularization strength used when computing the loss and gradient.
        r_seed: None or int.
            Random seed for controlling which samples are randomly included in each mini-batch during training.
        verbose: int.
            0 means no print outs. Any value > 0 prints Current epoch number and training loss every
            `print_every` (e.g. 100) epochs.
        print_every: int. How often (in epochs) to print loss and accuracy.
            If verbose > 0, print out the training loss and validation accuracy over the last epoch
            every `print_every` epochs.
            Example: If there are 20 epochs and `print_every` = 5 then you print-outs happen on
            after completing epochs 0, 5, 10, and 15 (or 1, 6, 11, and 16 if counting from 1).

        Returns:
        -----------
        loss_history: Python list of floats. len=`n_epochs * n_iter_per_epoch`.
            Recorded training loss for each mini-batch of training.
        train_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the training set.
        validation_acc_history: Python list of floats. len=`n_epochs`.
            Recorded accuracy on every epoch on the validation set.

        NOTE: Here is how accuracy and loss should be recorded.
        a) Record the accuracy:
            - on training set: Compute it on the ENTIRE training set only after completing an epoch.
            - on validation set: Compute it on the ENTIRE validation set only after completing an epoch.
        b) As in `SoftmaxLayer`, loss on training set should be recorded for each mini-batch of training.
        c) Every `print_every` iterations, print out (if `verbose` is `True`):
        '''
        rng = np.random.default_rng(r_seed)
        N = y.shape[0]
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        for epoch in range(n_epochs):
            # Shuffle indices for mini-batch sampling
            indices = np.arange(N)
            rng.shuffle(indices)
            for start_idx in range(0, N, mini_batch_sz):
                end_idx = start_idx + mini_batch_sz
                batch_idx = indices[start_idx:end_idx]
                X_batch = features[batch_idx]
                y_batch = y[batch_idx]
                # Forward Pass
                y_net_in, y_net_act, z_net_in, z_net_act, loss = self.forward(X_batch, y_batch, reg)
                loss_history.append(loss)
                # Backward Pass
                dy_wts, dy_b, dz_wts, dz_b = self.backward(X_batch, y_batch, y_net_in, y_net_act, z_net_in, z_net_act, reg)
                # Gradient Descent Update
                self.y_wts -= lr * dy_wts
                self.y_b -= lr * dy_b
                self.z_wts -= lr * dz_wts
                self.z_b -= lr * dz_b
            # Compute Accuracies after epoch
            train_preds = self.predict(features)
            val_preds = self.predict(x_validation)
            
            train_acc = self.accuracy(y, train_preds)
            val_acc = self.accuracy(y_validation, val_preds)
            
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            # Print progress 
            if verbose > 0 and (epoch % print_every == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
        return loss_history, train_acc_history, val_acc_history
