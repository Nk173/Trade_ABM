from init import case, countries, count, industries, P, A, alpha, beta, shock
from tqdm import tqdm

def gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, total_time = 3000, trade_time = 4000,
        partner_develops=False, pd_time=10000, shock=shock, capital_mobility=False, cm_time=6000, autarky_time=5000):
    
    from typing import Dict
    from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
    from functions import production_function, demand_function
    from tradeutils import doAllTrades
    from Agents import Nation
    import random
    import numpy as np
    import matplotlib.pyplot as plt 

    random.seed(0)

    ## 
    variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
    functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI','get_MRS']

    ## 
    inst = {}
    nationsdict={}
    for c in range(len(countries)):
        inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                A[countries[c]], alpha[countries[c]], beta[countries[c]],
                        pricing_algorithm=compute_price_marginal_utilities)
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

    # RCA container
    R_all={}
    for c in countries:
        R_all[c]={}
        for d in countries:
            R_all[c][d]=[]




    # Time Evolution of the Model
    for t in tqdm(range(total_time)):
        # Markers
        tr = False
        partner_develops = False
        capital_mobility = False

        if t>=trade_time:
            tr=True

        if t>=autarky_time:
            tr=False

        if t==pd_time:
            if partner_develops:
                for c in countries:
                    inst[c] = Nation(countries[c], count[c], industries, countries, P[countries[c]],  
                                     shock[countries[c]], alpha[countries[c]], beta[countries[c]],
                                     pricing_algorithm=compute_price_marginal_utilities)
                    nationsdict[countries[c]]=inst[c]

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
            nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)

            for v in range(len(variables)):
                for i in industries:
                    resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
            resultsU[c].append(nationsdict[c].get_utility())

    production = {}
    for c in countries:
        production[c] = nationsdict[c].production
        print(nationsdict[c].production)

    ## Plotting results
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

    plt.suptitle('Generalised {}'.format(case), fontsize=16)
    plt.savefig('Generalised/plots/{}_generalised_capital_mobility.png'.format(case))
    return production

# Model--Run
gulden(total_time=2000)

