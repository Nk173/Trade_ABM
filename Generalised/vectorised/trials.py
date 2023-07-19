import numpy as np
from agents_vec import gulden_vectorised as gv
import time
# Constants
iterations = 1000
n_countries = 3
countries = ['USA', 'China', 'India']
products= ['wine', 'cloth', 'wheat']
n_products = 3
alpha = np.array([[0.5, 0.5,0.5],
                  [0.5, 0.5,0.5],
                  [0.5,0.5,0.5]])  # output elasticity of labor
beta = np.array([[0.5, 0.5,0.5],
                 [0.5, 0.5,0.5],
                 [0.5, 0.5, 0.5]])  # output elasticity of capital
A = np.array([[2, 1,1],
              [1,2,1],
              [1,1,2]])  # Total Factor Productivity
shock = np.array([[0.5, 2.0],
                  [0.2, 0.8]])  # Total Factor Productivity

# Number of citizens in each nation
citizens_per_nation = [100, 100, 100]
t1 = time.time()
gv(n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
                  iterations=2000, Tr_time=500, pricing_algorithm='cpmu', utility_algorithm='geometric',
                  cap_mobility = True, shock =shock, shock_time=10000, cm_time=10000, plot=False)
t2 = time.time()
print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))