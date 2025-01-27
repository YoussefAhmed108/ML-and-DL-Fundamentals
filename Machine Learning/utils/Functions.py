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

def gini_index(values , mean = None):
    """
    Returns the gini impurity of the input array.
    
    Parameters
    ----------
    values : np.ndarray
        The input array.
    mean : float
        The mean value of the input array.
    
    Returns
    -------
    float
        The gini impurity of the input array.
    """
    if mean is None:
        n = len(values)
        if n == 0:
            return 0
        return 1 - np.sum([(np.sum(values == c) / n) ** 2 for c in np.unique(values)])
    else:
        return np.sum((values - mean) ** 2)

def entropy(values):
    """
    Returns the entropy of the input array.
    
    Parameters
    ----------
    values : np.ndarray
        The input array.
    
    Returns
    -------
    float
        The entropy of the input array.
    """
    n = len(values)
    if n == 0:
        return 0
    return -np.sum([(np.sum(values == c) / n) * np.log2(np.sum(values == c) / n) for c in np.unique(values)])
   