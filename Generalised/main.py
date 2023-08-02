from init import case, countries, count, industries, P, A, alpha, beta, shock, weights, elasticities, sigma
from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
from pricing import gd_pricing, demand_gap_pricing
from wages import wagesAsShareOfMarginalProduct, wageAsMarginalProductROIAsResidual, wageAsShareOfProduct
import time
def gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, 
           alpha=alpha, beta=beta, total_time = 3000, trade_time = 4000,
           pd_time=10000, shock=shock, cm_time=6000, autarky_time=5000,
           pricing_algorithm =compute_price_marginal_utilities, wage_algorithm = wageAsShareOfProduct,
           utility_algorithm = 'geometric', weights=weights,elasticities=elasticities,sigma=sigma, plot=True, csv=False, d=0):
    
    from typing import Dict
    from functions import production_function, demand_function
    from tradeutils import doAllTrades
    from Agents import Nation
    import random
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm 

    ## Set seed
    random.seed(0)

    ## Variables of Interest to Output from the model
    variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
    functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI','get_MRS']

    if pricing_algorithm==demand_gap_pricing:
        variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI']
        functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI']

    ## 
    inst = {}
    nationsdict={}
    for c in range(len(countries)):
        inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                A[countries[c]], alpha[countries[c]], beta[countries[c]],
                        pricing_algorithm=pricing_algorithm,
                        utility_algorithm=utility_algorithm,
                        wage_algorithm = wage_algorithm,d=d)
        nationsdict[countries[c]]=inst[c]

    # Results Container
    resultsdict = {}
    for c in countries:
        resultsdict[c] = {}
        for v in variables:
            resultsdict[c][v]={}
            for i in industries:
                resultsdict[c][v][i] = []

    # Trade Volume Dictionary
    tv: Dict[str, Dict[str, Dict[str, float]]] = {}
    for i in industries:
        tv[i] = {}
        for c in countries:
            tv[i][c]={}
            for n in countries:
                tv[i][c][n]=0.0


    # Utility Results Container
    resultsU = {}
    for c in countries:
        resultsU[c] = []

    try:
        if len(trade_time)>1:
            T0 = trade_time[0]
            T1 = trade_time[1]
    except:
        T0 = trade_time
        T1 = 0

# Time Evolution of the Model
    for t in tqdm(range(total_time)):
        # Markers
        tr = False
        partner_develops = False
        capital_mobility = False

        if t>=T0 or (T1>0 and t>=T1) :
            tr=True

        if t>=autarky_time:
            if (T1>0 and t<T1):
                tr=False

        if t==pd_time:
            partner_develops=True
            nationsdict[countries[1]].A = shock

        if t>=cm_time:
            capital_mobility=True

        for c in countries:
            nationsdict[c].update(nationsdict=nationsdict, capital_mobility=capital_mobility)

        if tr:
            trades = doAllTrades(tv, industries, countries, nationsdict, "wine")
            #tv = trades["trade_volume"] not needed, this is exported anyway...
            net_exports=trades["net_exports"]
            # print("t is " + str(t))
            # print(tv["wheat"]["USA"]["INDIA"])

        ## update trading....
        for c in countries:
            nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None, weights=weights, elasticities=elasticities, sigma=sigma)

            for v in range(len(variables)):
                for i in industries:
                    resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
                    
            resultsU[c].append(nationsdict[c].get_utility())

    # save results from resultsdict to a file
    if csv:
        import pandas as pd
        for nation in range(len(countries)):
            for industry in range(len(industries)):
                df = pd.DataFrame({'t':range(2000),
                                'labor':resultsdict[countries[nation]]['labor'][industries[industry]],
                                'capital':resultsdict[countries[nation]]['capital'][industries[industry]],
                                'production':resultsdict[countries[nation]]['production'][industries[industry]],
                                'wages':resultsdict[countries[nation]]['wages'][industries[industry]],
                                'ROI':resultsdict[countries[nation]]['ROI'][industries[industry]],
                                'prices':resultsdict[countries[nation]]['prices'][industries[industry]],
                                'demand':resultsdict[countries[nation]]['demand'][industries[industry]]
                                })
                df.to_csv('csvs/Model_{}_{}.csv'.format(countries[nation], industries[industry]))
    production={}
    for c in countries:
        production[c] = resultsdict[c]['production']

    ## Plotting results
    if plot == True:
        fig, axs = plt.subplots(5,2, figsize=(30, 30), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .5, wspace=.1)
        axs = axs.ravel()

        
        for c in countries:
            for k in range(len(variables)):
                for i in industries:
                    axs[k].plot(resultsdict[c][variables[k]][i], label='{}-{}-{}'.format(c, variables[k], i))
                    axs[k].set_title('{}'.format(variables[k]))
                    axs[k].legend(prop={'size': 10})
                    
            for k in range(len(variables), len(variables)+1):
                axs[k].plot(resultsU[c], label='{}-{}'.format(c,'utility'))
                axs[k].legend(prop={'size': 10})
                axs[k].set_title('{}'.format('Utility'))
        
        # axs[k].plot(sum(resultsU.values()), label='{}-{}'.format('World','utility'))
        plt.suptitle('Generalised {}'.format(case), fontsize=16)
        plt.savefig('plots/{}_wp.png'.format(case))
    return production, resultsdict

# Model--Run
t1 = time.time()
production, resultsdict = gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, total_time = 2000, trade_time = 500,
                                         pd_time=1000, shock=shock, cm_time=5000, autarky_time=5000,
                                         pricing_algorithm =demand_gap_pricing, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', 
                                         weights=weights, elasticities=elasticities, sigma=sigma, d=0.00000000, plot=False, csv=False)
t2 = time.time()
print('Trade_model Time taken: {} seconds'.format(t2-t1))