import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_train):
        score = X[i].dot(W)
        #logc = -1 * np.max(score)
        #score += logc
        loss += -score[y[i]] + np.log(np.sum(np.exp(score)))
        for j in range(num_class):
            logc = -1 * np.max(X[i].dot(W[:,j]))
            softjup = np.exp(X[i].dot(W[:,j]) + logc)
            softjdown = np.sum(np.exp(X[i].dot(W) + logc))
            softj = softjup / softjdown * X[i].T
            if j == y[i]:
                dW[:,j] += -X[i].T
                dW[:,j] += softj
                continue
            dW[:,j] += softj
     
  loss /= num_train
  loss += reg * np.sum(W*W)
  dW /= num_train
  dW += reg * 2 * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
  softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)
  #print(softmax_output.shape)
  loss += -1 * np.sum(np.log(softmax_output[np.arange(num_train), y]))
  loss /= num_train
  loss += reg * np.sum(W*W)
    
  softmax_output[np.arange(num_train), y] += -1
  dW = X.T.dot(softmax_output)
  dW /= num_train
  dW += reg * 2 * W
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

