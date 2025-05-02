import numpy as np
import pandas as pd
from scipy.optimize import minimize 
from scipy.optimize import differential_evolution
import scipy.stats as stats
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os, sys
import nevergrad as ng

import random
import sys, time
from functions import production_function, regularise, wage_function, demand_function, innovate, generate_nested_matrix, visualize_multi_layer_trade_network, array_to_dataframe
from tradeutils import doAllTrades
from pricing import updatePricesAndConsume
from agents_vec import gulden_vectorised
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# np.random.seed(0)

def opt_test(i):
    
    # Define the parameter space
    parametrization = ng.p.Array(shape=(9,)).set_bounds(0, 1)
    target_data = pd.read_csv('/rds/general/user/nk821/home/anaconda3/CCS/ABM/Gulden/vectorised_tests/io_cc_rand3val.csv', index_col=[0,1]).values

    # Define the optimizers to be compared
    optimizers_to_test = ['RandomSearch', 'NGOpt', 'DE', 'PSO', 'CMA', 'OnePlusOne','MultiBFGS','SVMMetaModel','HyperOpt','NelderMead']
    n_iter = 20000  # Number of iterations for each optimization

    # This dictionary will hold our optimization results
    optimization_results = {optimizer: [] for optimizer in optimizers_to_test}
    best_A_matrices = {optimizer: None for optimizer in optimizers_to_test}
    all_A_matrices = {optimizer: [] for optimizer in optimizers_to_test}

    def objective_function(A_flat):
        # Reshape A_values to matrix form
        A = np.reshape(A_flat, (3, 3))  # Reshape the flat array to a matrix

    #     A_values = [kwargs[f'a{i}{j}'] for i in range(3) for j in range(3)]
    #     A = np.reshape(A_values, (3, 3))    
        # Run the model
        Q, income, net_exports, io_df, io_df2, io_dfval, io_df2val = gulden_vectorised(case='test_3', n_countries=3, n_products=3, countries=['0', '1', '2'], 
                                                                                       products=['0', '1', '2'], citizens_per_nation=[100, 100, 100], A=A, 
                                                                                       alpha=np.ones((3, 3)) * 0.5, beta=np.ones((3, 3)) * 0.5, share=np.ones(3), 
                                                                                       shock=None, shock_time=10000, iterations=2000, Tr_time=1, cm_time=1, 
                                                                                       trade_change=0.01, autarky_time=15000, pricing_algorithm='cpmu', 
                                                                                       utility_algorithm='geometric', wage_algorithm='marginal_product', 
                                                                                       csv=False, plot=False)

        # Compute error
        error = np.mean((io_df2val.values - target_data) ** 2)
        return error

    # Run optimization for each selected method
#     for optimizer_name in optimizers_to_test:
    optimizer_name = optimizers_to_test[i-1]
    optimizer = ng.optimizers.registry[optimizer_name](parametrization=parametrization, budget=n_iter)
    best_so_far = np.inf
    best_A_so_far = None  # Initialize the best A matrix for this optimizer

    for _ in range(n_iter):
            x = optimizer.ask()
            value = objective_function(x.value)
            optimizer.tell(x, value)
            best_so_far = min(best_so_far, value)
            best_A_so_far = np.reshape(x.value, (3, 3))
            optimization_results[optimizer_name].append(best_so_far)

            # Save every A matrix instead of just the best one
            all_A_matrices[optimizer_name].append(np.reshape(x.value, (3, 3)))

    best_A_matrices[optimizer_name] = best_A_so_far


#     # Plot the convergence for each optimizer
#     fig, ax = plt.figure(figsize=(12, 8))
#     for optimizer_name, results in optimization_results.items():
#         plt.plot(results, label=optimizer_name)

#     plt.xlabel('Iteration', fontsize=14)
#     plt.ylabel('Best Objective Value', fontsize=14)
#     plt.title('Optimization Method Convergence Comparison', fontsize=16)
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#     fig.savefig('/rds/general/user/nk821/home/anaconda3/CCS/ABM/Gulden/vectorised_tests/comparison3x3.png', bbox_inches='tight')

    # Save optimization results and best A matrices to pickle files
    with open('/rds/general/user/nk821/home/anaconda3/CCS/ABM/Gulden/vectorised_tests/run2/optimization_results_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(optimization_results, f)

    with open('/rds/general/user/nk821/home/anaconda3/CCS/ABM/Gulden/vectorised_tests/run2/all_A_matrices_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(all_A_matrices, f)

    with open('/rds/general/user/nk821/home/anaconda3/CCS/ABM/Gulden/vectorised_tests/run2/best_A_matrices_{}.pkl'.format(i), 'wb') as f:
        pickle.dump(best_A_matrices, f)

if __name__=='__main__':
    
    i=int(sys.argv[1])
    opt_test(i)