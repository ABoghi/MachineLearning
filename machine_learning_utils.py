import numpy as np
import random as rm
import matplotlib
from matplotlib import pyplot as plt
import warnings
from typing import Union
from scipy.special import binom
import time


def sigmoid(z: float) -> float:
    '''
    Parameters
    ----------
    z: float
        sigmoid variable.

    Returns
    -------
    h: float
        regression.
    '''

    return 1.0/(1.0 + np.exp(-z))


def anti_sigmoid(h: float) -> float:
    '''
    Parameters
    ----------
    h: float
        regression. 

    Returns
    -------
    z: float 
        sigmoid variable.
    '''
    return np.log(h) - np.log(1.0 - h)


def verify_length(y: np.ndarray, h: np.ndarray) -> int:
    '''
    Parameters
    ----------
    y: np.ndarray
        data. 
    h: np.ndarray 
        regression.

    Returns
    -------
    n: int
        data/regression length.
    '''
    n = len(y)
    m = len(h)
    if n != m:
        quit()

    return n


def regularized_cost_function(J: float, theta: np.ndarray, Lambda: float, i_r: int) -> float:
    '''
    Parameters
    ----------
    J: float
        Cost Function.
    theta: np.ndarray
        Model Parameters.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    J: float
        Cost Function.
    '''

    m = len(theta)
    if i_r < m:
        for j in range(i_r, m):
            J += Lambda * pow(theta[j], 2.0) / (2.0 * m)

    return J


def regularized_cost_function_gradient(gradJ: np.ndarray, theta: np.ndarray, Lambda: float, i_r: int) -> np.ndarray:
    '''
    Parameters
    ----------
    gradJ: np.ndarray 
        Cost Function Gradient.
    theta: np.ndarray
        Model Parameters.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    gradJ: np.ndarray 
        Cost Function Gradient.
    '''
    m = len(theta)

    if i_r < m:
        for j in range(i_r, m):
            gradJ[j] += Lambda * theta[j] / m

    return gradJ


def linear_cost_function(y: np.ndarray, h: np.ndarray, theta: np.ndarray, Lambda: float, i_r: int) -> float:
    '''
    Parameters
    ----------
    y: np.ndarray
        data.  
    h: np.ndarray 
        regression.
    theta: np.ndarray
        Model Parameters.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    J: float
        Cost Function.
    '''

    n = len(y)

    J = 0.0
    for i in range(n):
        J += pow(h[i]-y[i], 2.0)/(2.0*n)

    J = regularized_cost_function(J, theta, Lambda, i_r)

    return J


def logistic_cost_function(y: np.ndarray, h: np.ndarray, theta: np.ndarray, Lambda: float, i_r: int) -> float:
    '''
    Parameters
    ----------
    y: np.ndarray
        data.  
    h: np.ndarray 
        regression.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    J: float
        Cost Function.
    '''

    n = len(y)

    J = 0.0
    for i in range(n):
        J -= (y[i] * np.log(h[i]) +
              (1.0 - y[i]) * np.log(1.0 - h[i]))/n

    J = regularized_cost_function(J, theta, Lambda, i_r)

    return J


def polynomial_terms(x: Union[np.ndarray, float], j: int) -> float:
    '''
    Parameters
    ----------
    x: Union[np.ndarray, float] 
        features.
    j: int
        index of the parameter theta.

    Returns
    -------
    term: float 
        polynomial term.
    '''
    if isinstance(x, float):
        term = pow(x, j)
    else:
        order = j
        m = len(x)
        nu = np.zeros(m)
        '''for i in range(m):
            nu[i] = order - i # binom(order-i, m)'''
        nu = find_polynomial_expansion_exponents(j, m)
        term = 1
        for i in range(m):
            term *= pow(x[i], nu[i])

    return term


def polynomial_model(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    '''
    Parameters
    ----------
    x: np.ndarray 
        features.
    theta: np.ndarray
        Model Parameters.

    Returns
    -------
    h: np.ndarray 
        regression.
    '''
    n = x.shape[1]

    m = len(theta)

    h = np.zeros(n)
    for i in range(n):
        for j in range(m):
            h[i] = h[i] + theta[j] * polynomial_terms(x[:, i], j)

    return h


def sigmoidal_model(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    '''
    Parameters
    ----------
    x: np.ndarray 
        features.
    theta: np.ndarray
        Model Parameters.

    Returns
    -------
    h: np.ndarray 
        regression.
    '''
    n = x.shape[1]

    z = polynomial_model(x, theta)

    h = np.zeros(n)
    for i in range(n):
        h[i] = sigmoid(z[i])

    return h


def cost_function_gradient(x: np.ndarray, y: np.ndarray, h: np.ndarray, theta: np.ndarray, Lambda: float, i_r: int) -> np.ndarray:
    '''
    Parameters
    ----------
    x: np.ndarray 
        features.
    y: np.ndarray
        data.  
    h: np.ndarray 
        regression.
    theta: np.ndarray
        Model Parameters.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    gradJ: np.ndarray 
        Cost Function Gradient.
    '''

    n = x.shape[1]
    m = len(theta)

    gradJ = np.zeros(m)
    for j in range(m):
        for i in range(n):
            gradJ[j] += ((h[i]-y[i])/n) * \
                polynomial_terms(x[:, i], j)

    gradJ = regularized_cost_function_gradient(gradJ, theta, Lambda, i_r)

    return gradJ


def run_gradient_descent(x: np.ndarray, y: np.ndarray, theta: np.ndarray, n_it: int, alpha: float, model: str, Lambda: float = 0.0, i_r: str = 3) -> np.ndarray:
    '''
    Parameters
    ----------
    x: np.ndarray 
        features.
    y: np.ndarray
        data. 
    theta: np.ndarray
        Model Parameters.
    n_it: int 
        number of iterations.
    alpha: float 
        learning rate.
    model: str 
        string containing the model.
    Lambda: float
        Regularization term.
    i_r: int
        Regularization index.

    Returns
    -------
    h: np.ndarray 
        regression.
    '''
    t_start = time.time()
    x_raws = x.shape[0]
    x_cols = x.shape[1]

    #n = verify_length(y, x)
    m = len(theta)
    if i_r >= m:
        warnings.warn(
            "Regularization index bigger than the number of parameter. No regularization performed")
    else:
        print(f"Regularization parameter = {Lambda}")

    alpha = abs(alpha)

    print("Learning Rate = "+str(alpha))

    Jv = []
    it = []
    # Normalize the feature
    Dx = np.zeros(x_raws)
    for i in range(x_raws):
        Dx[i] = np.amax(x[i, :]) - np.amin(x[i, :])
    x = x / Dx
    h, J = regression_model(model, x, y, theta, Lambda, i_r)
    J_old = J
    theta_old = theta
    for k in range(n_it):

        h, J = regression_model(model, x, y, theta, Lambda, i_r)

        gradJ = cost_function_gradient(x, y, h, theta, Lambda, i_r)
        for j in range(m):
            theta[j] = theta[j] - alpha * gradJ[j]
        Jv.append(J)
        it.append(k)
        #print("iteration = "+str(k)+"; Cost function = "+str(J))
        incr = 1.5
        if J < J_old:
            alpha = incr * alpha
        elif J > J_old:
            theta = theta_old
            alpha = alpha / incr
        else:
            alpha = alpha
        #print("alpha = "+str(alpha)+"; J = "+str(J))
        J_old = J
        theta_old = theta
    # Un-normalize the feature
    x = x * Dx
    for j in range(m):
        theta[j] = theta[j] / polynomial_terms(Dx, j)

    h, J = regression_model(model, x, y, theta, Lambda)

    generate_figures(it, Jv, x, y, h)

    print("Final Parameters:")
    print(theta)

    print("Final Learning Rate:")
    print(alpha)

    print(f"Execution time: {time.time() - t_start}[s]")

    return h


def generate_figures(it: np.ndarray, Jv: np.ndarray, x: np.ndarray, y: np.ndarray, h: np.ndarray):
    '''
    Parameters
    ----------
    it: np.ndarray
        iterations.
    Jv: np.ndarray 
        cost function per iteration.
    x: np.ndarray 
        features.
    y: np.ndarray
        data. 
    h: np.ndarray 
        regression.

    Returns
    -------
    '''

    plt.figure(1)
    plt.semilogy(it, Jv/Jv[0], 'r')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Normalized Cost Function")
    plt.figure(2)
    plt.plot(x[0][:], y, 'o', label='data')
    plt.plot(x[0][:], h, 'r', label='regression')
    plt.xlabel("Feature")
    plt.ylabel("y")
    plt.legend(loc='lower left')

    return


def artificial_feature(x_max: float, x_min: float, n: int) -> np.ndarray:
    '''
    Parameters
    ----------
    x_min: float 
        minimum value of the feature.
    x_max: float 
        maximum value of the feature.
    n: int
        feature array length

    Returns
    -------
    x: np.ndarray 
        features.
    '''
    dx = (x_max-x_min)/(n-1)
    x = np.zeros((1, n))
    for i in range(n):
        x[0][i] = x_min + dx * i

    return x


def polynomial_data_generator(theta: np.ndarray, x_min: float, x_max: float, sigma: float, n: int) -> np.ndarray:
    '''
    Parameters
    ----------
    theta: np.ndarray
        Model Parameters.
    x_min: float 
        minimum value of the feature.
    x_max: float 
        maximum value of the feature.
    sigma: float
        noise amplitude.
    n: int
        feature array length

    Returns
    -------
    x: np.ndarray 
        features.    
    y: np.ndarray
        data. 
    '''

    x = artificial_feature(x_max, x_min, n)

    y = polynomial_model(x, theta)

    for i in range(n):
        y[i] = y[i] + sigma * rm.uniform(-1, 1) / 2.0

    return x, y


def logistic_data_generator(theta: np.ndarray, x_min: float, x_max: float, sigma: float, n: int) -> np.ndarray:
    '''
    Parameters
    ----------
    theta: np.ndarray
        Model Parameters.
    x_min: float 
        minimum value of the feature.
    x_max: float 
        maximum value of the feature.
    sigma: float
        noise amplitude.
    n: int
        feature array length

    Returns
    -------
    x: np.ndarray 
        features.    
    y: np.ndarray
        data. 
    '''

    x = artificial_feature(x_max, x_min, n)

    z = polynomial_model(x, theta)

    y = np.zeros(n)
    for i in range(n):
        y[i] = sigmoid(z[i])

    if sigma > 1.0:
        sigma = 1.0
    elif sigma < 0.0:
        sigma = 0.0

    for i in range(n):
        y[i] = (1 - sigma) * y[i] + sigma * rm.uniform(0, 1)
        y[i] = rebound_between_zero_and_one(y[i])

    return x, y


def regression_model(model: str, x: np.ndarray, y: np.ndarray, theta: np.ndarray, Lambda: float = 0.0, i_r: int = 3) -> Union[np.ndarray, float]:
    '''
    Parameters
    ----------
    model: str 
        string containing the model.
    x: np.ndarray 
        features.
    y: np.ndarray
        data.  
    theta: np.ndarray
        Model Parameters.

    Returns
    -------
    h: np.ndarray 
        regression.
    J: float
        Cost Function.
    '''

    if model == 'linear':
        h = polynomial_model(x, theta)
        J = linear_cost_function(y, h, theta, Lambda, i_r)
    elif model == 'logistic':
        h = sigmoidal_model(x, theta)
        J = logistic_cost_function(y, h, theta, Lambda, i_r)
    else:
        raise ValueError('Unknown Regression Model')

    return h, J


def rebound_between_zero_and_one(y: float) -> float:
    '''
    Parameters
    ----------
    y: float 
        value to rebound.

    Returns
    -------
    y: float 
        bounded value.
    '''

    if y <= 0.5:
        y = 0.0
    else:
        y = 1.0

    return y


def find_polynomial_expansion_exponents(j: int, m: int) -> Union[float, np.ndarray]:
    """
    Parameters
    ----------
    j: int 
        position in the array indexing >= polynomial order
    m: int 
        number of variables

    Returns
    -------
    out: Union[float, np.ndarray] 
        polynomial exponents.
    """
    if j != 0:
        nu = np.array([])
        for i in polynomial_expansion_exponents(j, m):
            nu = np.append(nu, i)
        nu = np.reshape(nu, (-1, m))
        nu = np.insert(nu, 0, [0] * nu.shape[1], axis=0)
        out = nu[j]
    else:
        out = np.array([0])

    return out
# NOTE: this function has been adapted from Stack-Overflow. It would need further editing in the future.


def polynomial_expansion_exponents(order: int, n_variables: int):
    """
    Parameters
    ----------
    order: int 
        polynomial order
    n_variables: int 
        number of variables

    """

    pattern = [0] * n_variables
    for current_sum in range(1, order+1):
        pattern[0] = current_sum
        yield tuple(pattern)
        while pattern[-1] < current_sum:
            for i in range(2, n_variables + 1):
                if 0 < pattern[n_variables - i]:
                    pattern[n_variables - i] -= 1
                    if 2 < i:
                        pattern[n_variables - i + 1] = 1 + pattern[-1]
                        pattern[-1] = 0
                    else:
                        pattern[-1] += 1
                    break
            yield tuple(pattern)
        pattern[-1] = 0
