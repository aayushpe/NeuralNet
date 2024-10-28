import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture is:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two-layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The standard deviation for weight initialization
        """
        # First layer weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.b1 = np.zeros(hidden_dim)

        # Second layer weights and biases
        self.W2 = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.b2 = np.zeros(num_classes)

    def parameters(self):
        params = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }
        
        return params

    def forward(self, X):
        scores, cache = None, None

        # First fully-connected layer
        out_fc1, cache_fc1 = fc_forward(X, self.W1, self.b1)

        # ReLU activation
        out_relu, cache_relu = relu_forward(out_fc1)

        # Second fully-connected layer (output layer)
        scores, cache_fc2 = fc_forward(out_relu, self.W2, self.b2)

        # Store caches for backward pass
        cache = (cache_fc1, cache_relu, cache_fc2)

        return scores, cache

    def backward(self, grad_scores, cache):
        grads = None

        # Ufrom forward pass
        cache_fc1, cache_relu, cache_fc2 = cache

        # Backward pass 
        grad_out_relu, grad_W2, grad_b2 = fc_backward(grad_scores, cache_fc2)

        # Backward pass through ReLU 
        grad_out_fc1 = relu_backward(grad_out_relu, cache_relu)

        # Backward pass through layer
        _, grad_W1, grad_b1 = fc_backward(grad_out_fc1, cache_fc1)

        # Store gradients in a dictionary
        grads = {
            'W1': grad_W1,
            'b1': grad_b1,
            'W2': grad_W2,
            'b2': grad_b2
        }
        
        return grads
