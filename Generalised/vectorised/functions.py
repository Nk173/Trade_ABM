# Functions
import numpy as np

def production_function(A, L, K, alpha, beta):
    return  A * ((L)**(alpha)) * ((K)**(beta))

def wage_function(A, L, K, alpha,beta,p, 
                  p_function=production_function,
                  algorithm='marginal_product',
                  share=None):
    if algorithm=='marginal_product':
        inc_labor = L + 1
        inc_production = p_function(A, inc_labor, K, alpha, beta)
        Q = p_function(A, L, K, alpha, beta)
        wage = p*(inc_production - Q)
        roi = ((Q*p) - (wage*L))/K
    
    elif algorithm=='share_of_product':
        Q = p_function(A, L, K, alpha, beta)
        wage = share* p * Q/L
        roi = (1-share) * p * Q/K

    elif algorithm=='share_of_marginal_product':
        Q = p_function(A, L, K, alpha, beta)
        inc_labor = L + 1
        inc_production = p_function(A, inc_labor, K, alpha, beta)
        wage = share*p*(inc_production - Q)
        roi = ((Q*p) - (wage*L))/K

    return wage, roi

def demand_function(Y, P):
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (len(P) * P[p])

    return D