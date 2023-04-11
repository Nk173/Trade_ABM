# Initialisations
from init import countries, count, industries, P, A, alpha, beta
from functions import production_function, demand_function, RCA
from Agents import Nation, Citizen
import random
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd

global countries, industries, development_shock

## 
variables = ['production', 'demand', 'traded', 'labor', 'capital', 'wages', 'prices', 'ROI']
functions = ['get_production', 'get_demand', 'get_traded', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI']

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
              A[countries[c]], alpha[countries[c]], beta[countries[c]])
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
tv={}
for c in countries:
    tv[c]={}
    for n in countries:
        tv[c][n]=0

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
for t in range(1000):  
    # print('step',t)
    tr = False
    partner_develops = False
    if t>=500:
        tr=True

    for c in countries:
        nationsdict[c].update(trade_volume = tv, trade=tr, nationsdict=nationsdict, capital_mobility = False, partner_develops = partner_develops)
        if t>=500:    
            tv = nationsdict[c].get_trade_volume()

        for v in range(len(variables)):
            for i in industries:
                resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])
        # print(nationsdict[c].get_utility)       
        resultsU[c].append(nationsdict[c].get_utility())

    
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

plt.suptitle('Generalised', fontsize=16)
fig.savefig('plots/2C2I_generalised.png')
