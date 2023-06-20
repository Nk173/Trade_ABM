from init import case, countries, count, industries, P, A, alpha, beta, shock
from tqdm import tqdm

def gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, partner_develops=False, shock=shock):
    from typing import Dict
    from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
    from functions import production_function, demand_function
    from tradeutils import doAllTrades
    from Agents import Nation
    import random
    import numpy as np
    import matplotlib.pyplot as plt 

    from pricing import gd_pricing
    from wages import wagesAsShareOfMarginalProduct, wageAsMarginalProductROIAsResidual, wageAsShareOfProduct

    random.seed(0)

## 
variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI', 'MRS']
functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI','get_MRS']

# ## Initialisations
# countries = ['USA','CHINA']
# count = [100, 1000]
# industries = ['wine','cloth']

# P={}
# P['USA'] = [1,1]
# P['CHINA'] = [1,1]

# A={}
# A['USA']= [0.5, 2]
# A['CHINA'] = [0.2, 0.05]

# alpha={}
# alpha['USA'] = [0.5,0.5]
# alpha['CHINA'] = [0.5, 0.5]

# beta={}
# beta['USA'] = [0.5, 0.5]
# beta['CHINA'] = [0.5, 0.5]

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
for t in range(5000):
    # print('step',t)
    tr = False
    partner_develops = False
    if t>=1000:
        tr=True

    for c in countries:
        nationsdict[c].update(nationsdict=nationsdict)
    if tr:
        trades = doAllTrades(tv, industries, countries, nationsdict, "wine")
        #tv = trades["trade_volume"] not needed, this is exported anyway...
        net_exports=trades["net_exports"]
        print("t is " + str(t))
        print(tv["wheat"]["USA"]["INDIA"])
        # print(net_exports)
        # print("USA")
        # print(nationsdict["USA"].production)
        # print(nationsdict["USA"].prices)
        # print("CHINA")
        # print(nationsdict["CHINA"].production)
        # print(nationsdict["CHINA"].prices)
    ## update trading....
    for c in countries:
        nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)
        # print(nationsdict["CHINA"].supply)

            for v in range(len(variables)):
                for i in industries:
                    resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
            resultsU[c].append(nationsdict[c].get_utility())


print(nationsdict["USA"].production)
print(nationsdict["CHINA"].production)
print(nationsdict["INDIA"].production)

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
    plt.savefig('Generalised/plots/{}_generalised_stable.png'.format(case))
    return exports

# Model--Run
gulden()

