from typing import Dict

from Agents import Nation
import matplotlib.pyplot as plt

from Generalised.tradeutils import doAllTrades

plt.rcParams["figure.figsize"] = 20, 10

# repatriation pct
global repatriation_pct
repatriation_pct = 0.06

global tv
global development_shock_wine
global development_shock_cloth

# Preparing Output dictionary
countries = ['USA', 'CHINA']
variables = ['production', 'demand', 'supply', 'labor', 'capital', 'wages', 'prices', 'ROI']
functions = ['get_production', 'get_demand', 'get_supply', 'get_labor', 'get_capital', 'get_wages', 'get_prices',
             'get_ROI']
industries = ['wine', 'cloth']

resultsdict = {}
for c in countries:
    resultsdict[c] = {}
    for v in variables:
        resultsdict[c][v] = {}
        for i in industries:
            resultsdict[c][v][i] = []

# Setup Samuelson run with autarky: 0-500, Trade:500-1000, and partner_develops:1000-2000

## Create Nations
names = ['USA', 'CHINA']
count = [100, 1000]
industries = ['wine', 'cloth']
P_price_wine = [1, 1]
P_price_cloth = [1, 1]
A_wine = [0.5, 0.2]
A_cloth = [2, 0.05]
alpha_wine = [0.5, 0.5]
alpha_cloth = [0.5, 0.5]
beta_wine = [0.5, 0.5]
beta_cloth = [0.5, 0.5]

USA_ = Nation(names[0], count[0], industries,countries,
              [P_price_wine[0], P_price_cloth[0]],
              [A_wine[0], A_cloth[0]],
              [alpha_wine[0], alpha_cloth[0]],
              [beta_wine[0], beta_cloth[0]])
CHINA_ = Nation(names[1], count[1], industries,countries,
                [P_price_wine[1], P_price_cloth[1]],
                [A_wine[1], A_cloth[1]],
                [ alpha_wine[1], alpha_cloth[1]],
                [beta_wine[1], beta_cloth[1]])

nationsdict : Dict[str,Nation] = {
    "USA": USA_,
    "CHINA": CHINA_
}

pd = False
tr = False

tv: Dict[str, Dict[str, Dict[str, float]]] = {}
for i in industries:
    tv[i] = {}
    for c in countries:
        tv[i][c]={}
        for n in countries:
            tv[i][c][n]=0.0

net_exports = None
dev_shock = [0,0]
for t in range(4000):
    if t > 500:
        tr = True
    if t > 1000:
        pd = False
        CHINA_.A = [0.2, 0.8]
        #dev_shock = [0.2, 0.8]
    for c in countries:
        nationsdict[c].update(nationsdict=nationsdict)
    if tr:
        trades = doAllTrades(tv, industries, countries, nationsdict, "wine")
        #tv = trades["trade_volume"] not needed, this is exported anyway...
        net_exports=trades["net_exports"]
    ## update trading....
    for c in countries:
        nationsdict[c].updatePricesAndConsume(trade=tr,country_export=net_exports[c] if tr else None)
    if t >= 500:
        print(tv)
    for c in countries:
        for v in range(len(variables)):
            for i in industries:
                resultsdict[c][variables[v]][i].append(getattr(nationsdict[c], functions[v])()[i])

fig, axs = plt.subplots(4, 2, figsize=(15, 20), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace=.5, wspace=.1)
axs = axs.ravel()


import matplotlib.cm as cm
num_pairs = len(countries) * len(industries)
colors = cm.get_cmap('Paired')
pair_to_int = {pair: i for i, pair in enumerate(sorted([c + industry for c in countries for industry in industries]))}

for iter,c in enumerate(countries):
    for k in range(len(variables)):
        for i in industries:
            pair_int = pair_to_int[c + i]

            axs[k].plot(resultsdict[c][variables[k]][i],
                        color=colors(pair_int),
                        label='{}-{}-{}'.format(c, variables[k], i))
            axs[k].legend()
fig.savefig('samuelsonrun.png')
