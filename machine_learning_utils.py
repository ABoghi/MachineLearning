import numpy as np

import random as rm

import matplotlib

from matplotlib import pyplot as plt

def sigmoid(z) :
    '''
    Inputs
    ------
    z []: (float) sigmoid variable.

    Outputs
    -------
    h []: (float) 1D array containing the regression.
    '''
    
    return 1.0/(1.0 + np.exp(-z))

def antiSigmoid(h) :
    '''
    Inputs
    ------
    h []: (float) 1D array containing the regression. 

    Outputs
    -------
    z []: (float) sigmoid variable.
    '''
    return np.log(h) - np.log(1.0 - h)

def verifyLen(y,h) :
    '''
    Inputs
    ------
    y []: (float) 1D array containing the data. 
    h []: (float) 1D array containing the regression.

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
    y []: (float) 1D array containing the data. 
    h []: (float) 1D array containing the regression.

    Outputs
    -------
    J []: (float) Cost Function.
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
    y []: (float) 1D array containing the data. 
    h []: (float) 1D array containing the regression.

    Outputs
    -------
    J []: (float) Cost Function.
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
    x []: (float) 1D array containing the feature. 
    theta []: (float) 1D array containing the model parameters.

    Outputs
    -------
    h []: (float) 1D array containing the regression.
    '''
    n = len(x)
    m = len(theta)

    h = np.zeros(n)
    
    for i in range(n) :
        for j in range(m) :
            h[i] = h[i] + theta[j] * pow(x[i],j)

    return h

def SigmoidalModel(x,theta) :
    
    '''
    Inputs
    ------
    x []: (float) 1D array containing the feature. 
    theta []: (float) 1D array containing the model parameters.

    Outputs
    -------
    h []: (float) 1D array containing the regression.
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

def CostGradient(x,y,h,theta) :
    '''
    Inputs
    ------
    x []: (float) 1D array containing the feature.
    y []: (float) 1D array containing the data. 
    h []: (float) 1D array containing the regression.
    theta []: (float) 1D array containing the model parameters.

    Outputs
    -------
    gradJ []: (float) Cost Function Gradient.
    '''

    n = verifyLen(y,h)
    m = len(theta)

    gradJ = np.zeros(m)
    for j in range(m) :
        for i in range(n) :
            gradJ[j] = gradJ[j] + ((h[i]-y[i])/n) * pow(x[i],j)

    return gradJ

def runLinearGradientDescent(x,y,theta,n_it,alpha) :
    '''
    Inputs
    ------
    x []: (float) 1D array containing the feature.
    y []: (float) 1D array containing the data. 
    theta []: (float) 1D array containing the model parameters.
    n_it []: (int) number of iterations.
    alpha []: (float) learning rate.

    Outputs
    -------
    h []: (float) 1D array containing the regression.
    '''
    n = verifyLen(y,x)
    m = len(theta)

    alpha = abs(alpha)

    print("Learning Rate = "+str(alpha))

    Jv = []
    it = []
    # Normalize the feature
    Dx = max(x) - min(x)
    x = x / Dx
    for k in range(n_it) :
        h = PolynomialModel(x,theta)
        J = LinearCostFunction(y,h)
        gradJ = CostGradient(x,y,h,theta)
        for j in range(m) :
            theta[j] = theta[j] - alpha * gradJ[j]
        Jv.append(J)
        it.append(k)
        #print("iteration = "+str(k)+"; Cost function = "+str(J))
    # Un-normalize the feature
    x = x * Dx
    for j in range(m) :
        theta[j] = theta[j] / pow(Dx,j)
    
    h = PolynomialModel(x,theta)

    GenFigures(it,Jv,x,y,h)

    print("Final Parameters:")
    print(theta)

    return h

def GenFigures(it,Jv,x,y,h) :
    '''
    Inputs
    ------
    it []: (int)
    Jv []: (float) 
    x []: (float) 1D array containing the feature.
    y []: (float) 1D array containing the data.
    h []: (float) 1D array containing the regression.

    Outputs
    -------
    '''

    plt.figure(1)
    plt.semilogy(it,Jv/Jv[0],'r')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Normalized Cost Function")
    plt.figure(2)
    plt.plot(x,y,'o', label='data')
    plt.plot(x,h,'r', label='regression')
    plt.xlabel("Feature")
    plt.ylabel("y")
    plt.legend(loc='lower left')

    return

def PolynomialDataGenerator(theta,x_min,x_max,sigma,n) :
    '''
    Inputs
    ------
    theta []: (float) 1D array containing the model parameters.
    x_min []: (float) minimum value of the feature.
    x_max []: (float) maximum value of the feature.
    sigma []: (float) noise amplitude.
    n []: (int)

    Outputs
    -------
    x []: (float) 1D array containing the feature. 
    y []: (float) 1D array containing the data.
    '''

    dx = (x_max-x_min)/(n-1)
    x = np.zeros(n)
    for i in range(n) :
        x[i] = x_min + dx * i

    y = PolynomialModel(x,theta)

    for i in range(n) :
        y[i] = y[i] + sigma * rm.uniform(-1,1) / 2.0

    return x, y