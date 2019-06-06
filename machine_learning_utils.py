import numpy as np

import random as rm

import matplotlib

from matplotlib import pyplot as plt

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

    return n

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

    n = verifyLen(y,h)

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

    n = verifyLen(y,h)

    J = 0.0
    for i in range(n) :
        J = J - ( y[i] * np.log(h[i]) + (1.0 - y[i]) * np.log(1.0 - h[i]) )/n

    return J

def PolynomialModel(x,theta) :
    
    '''
    Inputs
    ------
    x []: (float) 
    theta []: (float)

    Outputs
    -------
    h []: (float)
    '''
    n = len(x)
    m = len(theta)

    print(theta)
    print(m)

    h = np.zeros(n)
    for j in range(m) :
        print("j = "+str(j))
        for i in range(n) :
            h[i] = h[i] + theta[j] * pow(x[i],j)

    return h

def SigmoidalModel(x,theta) :
    
    '''
    Inputs
    ------
    x []: (float) 
    theta []: (float)

    Outputs
    -------
    h []: (float)
    '''
    n = len(x)
    m = len(theta)

    print(theta)
    print(m)

    z = np.zeros(n)
    for j in range(m) :
        print("j = "+str(j))
        for i in range(n) :
            z[i] = z[i] + theta[j] * pow(x[i],j)

    h = np.zeros(n)
    for i in range(n) :
        h[i] = sigmoid(z[i])

    return h

def LinearCostGradient(x,y,h,theta) :
    '''
    Inputs
    ------
    x []: (float)
    y []: (float) 
    h []: (float)
    theta []: (float)

    Outputs
    -------
    J []: (float)
    '''

    n = verifyLen(y,h)
    m = len(theta)

    
    gradh = np.zeros(m)
    for j in range(m) :
        for i in range(n) :
            gradh[j] = gradh[j] + pow(x[i],j)

    gradJ = np.zeros(m)
    for j in range(m) :
        for i in range(n) :
            gradJ[j] = gradJ[j] + ((h[i]-y[i])/n) * gradh[j]

    return gradJ

def LogisticCostGradient(x,y,h,theta) :
    '''
    Inputs
    ------
    x []: (float)
    y []: (float) 
    h []: (float)
    theta []: (float)

    Outputs
    -------
    J []: (float)
    '''

    n = verifyLen(y,h)
    m = len(theta)

    gradz = np.zeros(m)
    for j in range(m) :
        for i in range(n) :
            gradz[j] = gradz[j] + pow(x[i],j)

    gradJ = np.zeros(m)
    for j in range(m) :
        for i in range(n) :
            gradJ[j] = gradJ[j] + ( h[i] - y[i] ) / n * gradz[j]

    return gradJ