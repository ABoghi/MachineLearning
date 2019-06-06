import numpy as np

import random as rm

def sigmoid(z) :
    '''
    Inputs
    ------
    z []: (float) 

    Outputs
    -------
    h []: (float)
    '''
    
    return 1.0/(1.0 + np.exp(-z))

def antiSigmoid(h) :
    '''
    Inputs
    ------
    h []: (float) 

    Outputs
    -------
    z []: (float)
    '''
    return np.log(h) - np.log(1.0 - h)

def verifyLen(y,h) :
    '''
    Inputs
    ------
    y []: (float) 
    h []: (float)

    Outputs
    -------
    '''
    n = len(y)
    m = len(h)
    if n != m :
        quit()

    return

def LinearCostFunction(y,h) :
    '''
    Inputs
    ------
    y []: (float) 
    h []: (float)

    Outputs
    -------
    J []: (float)
    '''

    verifyLen(y,h)

    J = 0.0
    for i in range(n) :
        J = J + pow(h[i]-y[i],2.0)/(2.0*n)

    return J

def LogisticCostFunction(y,h) :

    '''
    Inputs
    ------
    y []: (float) 
    h []: (float)

    Outputs
    -------
    J []: (float)
    '''

    verifyLen(y,h)

    J = 0.0
    for i in range(n) :
        J = J - ( y[i] * np.log(h[i]) + (1.0 - y[i]) * np.log(1.0 - h[i]) )/n

    return J

def PolynomialModel(x,theta) :
    
    n = len(x)
    m = len(theta)

    h = np.zeros(n)
    for j in range(m) :
        for i in range(n) :
            h[i] = theta[j] * pow(x[i],j)

    return h