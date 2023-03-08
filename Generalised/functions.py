import random
import numpy as np
# Define the Cobb-Douglas production function
def production_function(A, alpha, L, beta, K):
    return A * (L ** alpha) * (K ** beta)

# Define the demand function
def demand_function(Y, P):
    
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (2 * P[p])

    return D

# Define a Comparative Advantage function
def RCA(A,B):
    R_a = np.zeros((len(A)))
    R_b = np.zeros((len(A)))
    Rab = np.divide(A,B)
    id_a = np.argmax(Rab)
    id_b = np.argmin(Rab)
    R_a[id_a] = 1
    R_b[id_b] = 1
    return(R_a)