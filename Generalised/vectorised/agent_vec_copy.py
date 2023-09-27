import numpy as np
import random
import sys, time, datetime, os
from functions import production_function, wage_function, demand_function, innovate
from tradeutils import doAllTrades
from pricing import updatePricesAndConsume
from tqdm import tqdm
import yaml
from matplotlib.cm import get_cmap
cmap = get_cmap('Set1')  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list


np.random.seed(1)


# Constants
case = '2x2'
iterations = 1000
n_countries = 2
countries = ['USA', 'China']
products= ['wine', 'cloth']
n_products = 2
alpha = np.array([[0.5, 0.5],
                  [0.5, 0.5]])  # output elasticity of labor

beta = np.array([[0.5, 0.5],
                 [0.5, 0.5]])  # output elasticity of capital

A = np.array([[1, 0.8],
              [1, 0.8]])  # Total Factor Productivity

shock = np.array([[0.5, 2.0],
                  [0.2, 0.8]])  # Total Factor Productivity

share = np.array([1,0.5])
# Number of citizens in each nation
citizens_per_nation = [500, 500]


def gulden_vectorised(case, n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta, 
                      iterations=3000, Tr_time=1000, trade_change=0.5, autarky_time= 10000, pricing_algorithm='cpmu', utility_algorithm='geometric', wage_algorithm = 'marginal_product', share=share,
                      csv = False, plot = False, yaml_dump=False, shock = shock, shock_time = 10000, cm_time=10000, d=0.0,
                      innovation = False, innovation_time = 2000, innovation_algorithm='investment_based', eta = 0.01, weights=None, elasticities=None, sigma=None):
    
    ## Citizen-level
    # Initialize prices for each good in each nation
    prices = np.ones((n_countries, n_products))
    prices_vec = np.ones((n_countries, n_products, iterations))

    # Initialize industry choices for labor and capital for each citizen in each nation
    labor_choices = [np.zeros(citizens, dtype=int) for citizens in citizens_per_nation]
    capital_choices = [np.zeros((citizens, 2), dtype=int) for citizens in citizens_per_nation]    
    income = [np.zeros(citizens) for citizens in citizens_per_nation]
    
    # Allow capital choices to take both industry and nation if capital mobility is enabled
    for nation in range(n_countries):
        labor_choices[nation] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:,0] = np.random.choice(range(n_products), citizens_per_nation[nation])
        capital_choices[nation][:,1] = np.ones(citizens_per_nation[nation])*nation

    ## Nation-level
    # Calculate wages and returns to capital for each industry in each nation
    L = np.zeros((n_countries, n_products))
    K = np.zeros((n_countries, n_products))
    Q = np.zeros((n_countries, n_products))
    W = np.zeros((n_countries, n_products))
    R = np.zeros((n_countries, n_products))
    D = np.zeros((n_countries, n_products))
    S = np.zeros((n_countries, n_products))
    net_exports = np.zeros(S.shape, dtype=np.float128)

    tv = np.zeros((n_countries, n_countries, n_products))
    labor = np.zeros((n_countries, n_products,iterations))
    capital = np.zeros((n_countries, n_products,iterations))
    production = np.zeros((n_countries, n_products,iterations))
    demand = np.zeros((n_countries, n_products,iterations))
    supply = np.zeros((n_countries, n_products,iterations))
    net_exports_history = np.zeros((n_countries, n_products,iterations))
    wage = np.zeros((n_countries, n_products,iterations))
    returns = np.zeros((n_countries, n_products,iterations))
    utility = np.zeros((n_countries, iterations))
    gdp_vec = np.zeros((n_countries, iterations))
    gnp_vec = np.zeros((n_countries, iterations))

    try:
        if len(Tr_time)>1:
            T0 = Tr_time[0]
            T1 = Tr_time[1]
    except:
        T0 = Tr_time
        T1 = 0

    for t in tqdm(range(iterations)):

        Tr=False
        cap_mobility = False

        if t>=T0 or (T1>0 and t>=T1) :
            Tr=True

        if t>=autarky_time:
            if (T1>0 and t<T1):
                Tr=False
                tv = np.zeros((n_countries, n_countries, n_products))

        if t>shock_time:
            A = shock

        if t>=cm_time:
            cap_mobility = True

        
        for nation in range(n_countries):
            # Initialise 
            should_change_labor = np.zeros(citizens_per_nation[nation])
            should_change_capital = np.zeros(citizens_per_nation[nation])

            # Citizen update
            # Re-assess labor and capital choices
            # Citizens decide whether to change their job or investment based on probability P
            max_wage_industry = np.argmax(W[nation,:])
            event_occurs = np.random.binomial(1, 0.004, size=citizens_per_nation[nation])
            should_change_labor = (event_occurs==1) * (W[nation, labor_choices[nation]]< W[nation, max_wage_industry]*(1+d))
            
            if cap_mobility:
                max_return_industry = np.unravel_index(np.argmax(R, axis=None), R.shape)
                should_change_capital = (event_occurs==1) * (R[capital_choices[nation][:,1], capital_choices[nation][:,0]]<R[max_return_industry]*(1+d))
                capital_choices[nation][should_change_capital==1] = (max_return_industry[1],max_return_industry[0])
            
            else:
                max_return_industry = np.argmax(R[nation,:])
                should_change_capital = (event_occurs==1) * (R[nation, capital_choices[nation][:,0]]< R[nation, max_return_industry]*(1+d))
                capital_choices[nation][should_change_capital==1] = (max_return_industry,nation)

            labor_choices[nation][should_change_labor==1] = max_wage_industry
            
            income[nation] = sum(W[nation, labor_choices[nation]] + R[capital_choices[nation][:,1], capital_choices[nation][:,0]])

            for industry in range(n_products):
                L[nation,industry] = 0 
                D[nation,industry] = income[nation]/(n_products*prices[nation,industry])
                demand[nation,industry,t] = D[nation,industry]
                L[nation,industry] = sum(labor_choices[nation] == industry)

                if cap_mobility:
                    K[nation, industry]=0
                    K[nation,industry] = sum((capital_choices[c][i,0]==industry) and (capital_choices[c][i,1]==nation) for c in range(n_countries) for i in range(citizens_per_nation[c]))
                
                else:
                    K[nation, industry]=0
                    K[nation,industry] = sum(capital_choices[nation][:,0] == industry)
                
                if K[nation, industry]<1:
                    K[nation, industry] = 1

            # Calculate labor, capital, production, wages, and returns for each industry in each nation
            # for industry in range(n_products):
                Q[nation,industry] = production_function(A[nation,industry], L[nation,industry], K[nation,industry],alpha[nation,industry], beta[nation,industry])
                W[nation, industry], R[nation, industry] =  wage_function(A[nation,industry], L[nation,industry], K[nation,industry], alpha[nation,industry], beta[nation,industry],p=prices[nation, industry],algorithm=wage_algorithm, share=share[nation])
                S[nation, industry] = Q[nation, industry]

                # Containers
                labor[nation,industry,t] = L[nation,industry]
                capital[nation,industry,t] = K[nation,industry]
                production[nation,industry,t] = Q[nation,industry]
                wage[nation,industry,t] = W[nation, industry]
                returns[nation,industry,t] = R[nation, industry]
                # supply[nation, industry,t] = S[nation, industry]
            gnp_vec[nation,t] = sum(Q[nation,:])

        # Trade update
        if Tr:
            tv, net_exports = doAllTrades(tv, S, prices, trade_change)
            # net_exports = trades[1]
            S = Q - net_exports

        net_exports_history[:,:,t] = net_exports

        # Update prices and Consume
        prices, UT = updatePricesAndConsume(prices, D, S, pricing_algorithm, utility_algorithm, weights=weights, elasticities=elasticities, sigma=sigma)

        prices_vec[:,:,t] = prices
        utility[:,t] = UT
        gdp_vec[:,t] = income
        supply[:,:,t] = S
        
        # Innovate
        if (innovation) and (t>=innovation_time):
            A = innovate(A,K, Q, method=innovation_algorithm, beta_lbd=0.01, eta=eta, beta_diffusion=0.02)

                
    # Save the results of Labor, Productions, Wages, and Returns to a csv file
    if csv:
        import pandas as pd
        for nation in range(n_countries):
            for industry in range(n_products):
                df = pd.DataFrame({'t':range(iterations),
                                'labor':labor[nation,industry,:],
                                'capital':capital[nation,industry,:],
                                'production':production[nation,industry,:], 
                                'wages':wage[nation,industry,:],  
                                'returns':returns[nation,industry,:],
                                'prices':prices_vec[nation,industry,:],
                                'demand':demand[nation,industry,:],
                                'utility':utility[nation,:]})
                df.to_csv('csvs/vectorised_{}_{}.csv'.format(countries[nation], products[industry]))
    
    # Plot the results
    if plot:
        import matplotlib.pyplot as plt
        # Define a custom color palette with the first four colors fixed
        custom_colors = ['black', 'yellow', 'orange', 'blue']
        remaining_colors = plt.cm.get_cmap('tab20', n_countries * n_products - 4)(range(n_countries * n_products - 4))
        color_palette = custom_colors + remaining_colors.tolist()
        # variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
        fig,ax = plt.subplots(6,2, figsize=(10,30))
        for nation in range(n_countries):
            for industry in range(n_products):
                color = color_palette[nation * n_products + industry]  
                ax[0,0].plot(range(iterations), production[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[0,0].set_title('Production')
                ax[0,1].plot(range(iterations), net_exports_history[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]),color=color)
                ax[0,1].set_title('Net Exports')

                ax[1,0].plot(range(iterations), labor[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[1,0].set_title('Labor')
                ax[1,1].plot(range(iterations), capital[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[1,1].set_title('Capital')

                ax[2,0].plot(range(iterations), wage[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[2,0].set_title('Wages')
                ax[2,1].plot(range(iterations), returns[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[2,1].set_title('Returns')

                ax[3,0].plot(range(iterations), prices_vec[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[3,0].set_title('Prices')
                ax[3,1].imshow(A)

                ax[5,1].plot(range(iterations), demand[nation, industry,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[5,1].set_title('Demand')
                ax[5,0].plot(range(iterations), supply[nation, industry, :], label='{}-{}'.format(countries[nation], products[industry]), color=color)
                ax[5,0].set_title('Supply')

            ax[4,0].plot(range(iterations), gdp_vec[nation,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
            ax[4,0].set_title('GDP')
            ax[4,1].plot(range(iterations), utility[nation,:], label='{}-{}'.format(countries[nation], products[industry]), color=color)
            ax[4,1].set_title('Utility')
        ax[3,0].legend()

        today = datetime.datetime.now()
        folder = today.strftime('%Y%m%d')
        # os.makedirs('plots/{}'.format(folder))
        plt.savefig('plots/{}/vectorised_{}.png'.format(folder, case))
        # plt.savefig('plots/vectorised_{}.png'.format(case))

        if yaml_dump:
            # Saving parameters
            def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
                return dumper.represent_list(array.tolist())

            yaml.add_representer(np.ndarray, ndarray_representer)
            params = {
                        'ProductionFunction': {
                        'alpha': alpha,
                        'beta': beta,
                        'A': A
                        },
                        'Options':{
                            'iterations':iterations,
                            'Trade':Tr_time,
                            'shock':shock_time,
                            'capital_mobility':cm_time,
                            'wage_share':share,
                            'innovation_time':innovation_time
                        },
                        'Algorithms':{
                            'pricing_algorithm':pricing_algorithm,
                            'utility_algorithm':utility_algorithm,
                            'wage_algorithm':wage_algorithm,
                            'innovation_algorithm':innovation_algorithm
                        },
                        'Constants':{
                            'eta':eta,
                            'd':d
                        }
                    }
            
            # Save parameters to a YAML file
            with open('plots/{}/params_{}.yaml'.format(folder,case), 'w') as file:
                    yaml.dump(params, file)

    return production

# model run
# t1 = time.time()
# gulden_vectorised('share_without_Tr', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=3000, Tr_time=10000, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]),
#                   shock = shock, shock_time=10000, cm_time=15000, plot=True, csv=False, yaml_dump=True, d=0.0,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

t1 = time.time()
gulden_vectorised('share_with_Tr', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
                  iterations=5000, Tr_time=1000, trade_change=0.5,autarky_time=1500000, pricing_algorithm='dgp', utility_algorithm='geometric',
                  wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]), cm_time=150000, shock_time=100000,
                  plot=True, csv=False, yaml_dump=True, d=0.05,  innovation_time = 300000,
                  innovation = False)

t2 = time.time()
print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# t1 = time.time()
# gulden_vectorised('share_with_Tr=0.5_ir', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=10000, Tr_time=1000, trade_chage=0.5, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]),
#                   shock = shock, shock_time=10000, cm_time=15000, plot=True, csv=False, yaml_dump=True, d=0.0,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# t1 = time.time()
# gulden_vectorised('share_with_Tr_and_cm_ir', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=5000, Tr_time=1000, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]),
#                   shock = shock, shock_time=10000, cm_time=3000, plot=True, csv=False, yaml_dump=True, d=0.0,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# # No trade with capital mobility
# t1 = time.time()
# gulden_vectorised('share_with_cm_dgp', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=5000, Tr_time=100000, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]),
#                   shock = shock, shock_time=10000, cm_time=3000, plot=True, csv=False, yaml_dump=True, d=0.0,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# t1 = time.time()
# gulden_vectorised('share_with_Tr_and_cm_increasing_returns', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=5000, Tr_time=1000, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([1,0.5]),
#                   shock = shock, shock_time=10000, cm_time=3000, plot=True, csv=False, yaml_dump=False, d=0.0,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))
# t1 = time.time()
# gulden_vectorised('share_product', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=2000, Tr_time=500, autarky_time=15000, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='share_of_marginal_product', share=np.array([0.5,1]),
#                   shock = shock, shock_time=10000, cm_time=1000, plot=True, csv=False, d=0.0)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))

# # GB
# n_countries = 2
# countries = ['USA', 'Ghana']
# products= ['wine', 'cloth']
# n_products = 2
# alpha = np.array([[0.7, 0.4],
#                   [0.7, 0.4]])  # output elasticity of labor

# beta = np.array([[0.7, 0.4],
#                  [0.7, 0.4]])  # output elasticity of capital

# A = np.array([[1, .5],
#               [0.2, 0.2]])  # Total Factor Productivity

# shock = np.array([[1, .5],
#                   [1.5, 0.2]])  # Total Factor Productivity

# # Number of citizens in each nation
# citizens_per_nation = [500, 100]
# t1 = time.time()
# gulden_vectorised('GB', n_countries, n_products, countries, products, citizens_per_nation, A, alpha, beta,
#                   iterations=3000, Tr_time=[500, 2000], autarky_time=1500, pricing_algorithm='dgp', utility_algorithm='geometric',
#                   wage_algorithm='marginal_product', share=np.array([1,1]),
#                   shock = shock, shock_time=1000, cm_time=5000, plot=True, csv=False, yaml_dump=True, d=0.0005,
#                   innovation = False, innovation_time = 30000, innovation_algorithm='learning_by_doing', eta = 0.1)
# t2 = time.time()
# print('Vectorised Trade model time taken: {} seconds'.format(t2-t1))




