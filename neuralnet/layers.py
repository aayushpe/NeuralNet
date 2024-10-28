import numpy as np


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.

    The input x has shape (N, Din) and contains a minibatch of N
    examples, where each example x[i] has shape (Din,).

    Inputs:
    - x: A numpy array of shape (N, Din) giving input data
    - w: A numpy array of shape (Din, Dout) giving weights
    - b: A numpy array of shape (Dout,) giving biases

    Returns a tuple of:
    - out: output, of shape (N, Dout)
    - cache: (x, w, b)
    """
    out = None

    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    x_reshape = x.reshape(row_dim, col_dim)
    out = np.dot(x_reshape, w) + b
    
    cache = (x, w, b)
    return out, cache


def fc_backward(grad_out, cache):
    """
    Computes the backward pass for a fully-connected layer.

    Inputs:
    - grad_out: Upstream derivative, of shape (N, Dout)
    - cache: Tuple of:
      - x: Input data, of shape (N, Din)
      - w: Weights, of shape (Din, Dout)
      - b: Biases, of shape (Dout,)

    Returns a tuple of:
    - grad_x: Gradient with respect to x, of shape (N, Din)
    - grad_w: Gradient with respect to w, of shape (Din, Dout)
    - grad_b: Gradient with respect to b, of shape (Dout,)
    """
    x, w, b = cache
    N = x.shape[0]
    x_reshaped = x.reshape(N, -1)

    grad_x_reshaped = np.dot(grad_out, w.T)         # Shape: (N, Din)
    grad_w = np.dot(x_reshaped.T, grad_out)         # Shape: (Din, Dout)
    grad_b = np.sum(grad_out, axis=0)               # Shape: (Dout,)

    grad_x = grad_x_reshaped.reshape(x.shape)       # Reshape back to input shape

    return grad_x, grad_w, grad_b


def relu_forward(x):
    """
    Computes the forward pass for the Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - x: A numpy array of inputs, of any shape

    Returns a tuple of:
    - out: A numpy array of outputs, of the same shape as x
    - cache: x
    """
    out = None

    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(grad_out, cache):
    """
    Computes the backward pass for a Rectified Linear Unit (ReLU) nonlinearity

    Input:
    - grad_out: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - grad_x: Gradient with respect to x
    """
    grad_x, x = None, cache

    out = np.maximum(0, x)
    out[out > 0] = 1
    grad_x = out * grad_out
    
    return grad_x


def l2_loss(x, y):
    """
    Computes the loss and gradient of L2 loss.

    loss = 0.5 * sum_i (x_i - y_i)**2 / N

    Inputs:
    - x: Input data, of shape (N, D)
    - y: Output data, of shape (N, D)

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    diff = x - y
    loss = 0.5 * np.sum(diff * diff) / N
    grad_x = diff / N
    return loss, grad_x


def softmax_loss(x, y):
    """
    Computes the loss and gradient for the softmax (cross-entropy) loss function.

    Inputs:
    - x: Numpy array of shape (N, C) giving predicted class scores, where
      x[i, c] gives the predicted score for class c on input sample i
    - y: Numpy array of shape (N,) giving ground-truth labels, where
      y[i] = c means that input sample i has ground truth label c, where
      0 <= c < C.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - grad_x: Numpy array of shape (N, C) giving the gradient of the loss with
      respect to x
    """
    loss, grad_x = None, None
    
    # Shift the logits for numerical stability
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    
    # Compute softmax probabilities
    probs = np.exp(shifted_logits)
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]

    # Compute the loss
    correct_logprobs = -np.log(probs[np.arange(N), y])
    loss = np.sum(correct_logprobs) / N
    
    # Compute the gradient
    grad_x = probs.copy()
    grad_x[np.arange(N), y] -= 1
    grad_x /= N
    
    return loss, grad_x


def l2_regularization(w, reg):
    """
    Computes loss and gradient for L2 regularization of a weight matrix:

    loss = (reg / 2) * sum_i w_i^2

    Where the sum ranges over all elements of w.

    Inputs:
    - w: Numpy array of any shape
    - reg: float giving the regularization strength

    Returns:
    - loss: Scalar giving the L2 regularization loss
    - grad_w: Numpy array of same shape as w giving gradient of loss with respect to w
    """
    # Compute the L2 loss
    loss = (reg / 2) * np.sum(w ** 2)
    
    # Compute the gradient of the loss with respect to w
    grad_w = reg * w
    
    return loss, grad_w
