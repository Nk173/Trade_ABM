import random
import numpy as np
import sys
from typing import Dict, List

# Define the Cobb-Douglas production function
def production_function(A, alpha, L, beta, K):
    return A * (L ** alpha) * (K ** beta)

# Define the demand function
def demand_function(Y, P):
    
    D = np.zeros((len(P)))
    for p in range(len(P)):
        D[p] = Y / (len(P) * P[p])

    return D