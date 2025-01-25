import numpy as np

def softmax(z):
    """
    Returns the softmax of the input array.
    
    Parameters
    ----------
    z : np.ndarray
        The input array.
    
    Returns
    -------
    np.ndarray
        The softmax of the input array.
    """
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def sigmoid(z):
    """
    Returns the sigmoid of the input array.
    
    Parameters
    ----------
    z : np.ndarray
        The input array.
    
    Returns
    -------
    np.ndarray
        The sigmoid of the input array.
    """
    return 1 / (1 + np.exp(-z))

def soft_threshold(rho, alpha):
    """
    Returns the soft threshold of the input array.
    
    Parameters
    ----------
    rho : np.ndarray
        The input array.
    alpha : float
        The threshold value.
    
    Returns
    -------
    np.ndarray
        The soft threshold of the input array.
    """
    return np.sign(rho) * np.maximum(np.abs(rho) - alpha, 0)