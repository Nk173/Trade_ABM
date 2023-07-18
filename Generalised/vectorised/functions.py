# Functions
import numpy as np

def production_function(A, L, K, alpha, beta):
    return  A * ((L)**(alpha)) * ((K)**(beta))

def wage_function(A, L, K, alpha,beta,p, p_function=production_function):
    inc_labor = L + 1
    inc_production = p_function(A, inc_labor, K, alpha, beta)
    wage = p*(inc_production - p_function(A, L, K, alpha, beta))
    return wage

def demand_function(Y, P):
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (len(P) * P[p])

    return D