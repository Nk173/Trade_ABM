from Agents import Nation, Citizen
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]=20,10

# repatriation pct
global repatriation_pct
repatriation_pct = 0.06

global tv
global development_shock_wine
global development_shock_cloth


# Preparing Output dictionary
countries = ['USA','CHINA']
variables = ['production', 'demand','supply','labor','capital','wages', 'prices','ROI']
functions = ['get_production', 'get_demand', 'get_supply', 'get_labor', 'get_capital', 'get_wages', 'get_prices', 'get_ROI']
industries = ['wine','cloth']

resultsdict = {}
for c in countries:
    resultsdict[c] = {}
    for v in variables:
        resultsdict[c][v]={}
        for i in industries:
            resultsdict[c][v][i] = []
            



# Setup Samuelson run with autarky: 0-500, Trade:500-1000, and partner_develops:1000-2000

## Create Nations
names = ['USA','CHINA']
count = [100,1000]
P_price_wine = [1,1]
P_price_cloth = [1,1]
A_wine = [0.5,0.2]
A_cloth = [2,0.05]
alpha_wine = [0.5, 0.5]
alpha_cloth = [0.5, 0.5]
beta_wine = [0.5, 0.5]
beta_cloth = [0.5, 0.5]
USA_ = Nation(names[0], count[0], P_price_wine[0], P_price_cloth[0], 
              A_wine[0], A_cloth[0], alpha_wine[0], alpha_cloth[0],
              beta_wine[0], beta_cloth[0])
CHINA_ = Nation(names[1], count[1], P_price_wine[1], P_price_cloth[1], 
              A_wine[1], A_cloth[1], alpha_wine[1], alpha_cloth[1],
              beta_wine[1], beta_cloth[1]) 

nationsdict={
    "USA":USA_,
    "CHINA":CHINA_
}


pd=False
tr=False
dev_shock = [0,0]
tv=0
for t in range(2000):  
    if t>500:
        tr = True
    if t>1000:
        pd = True
        dev_shock = [0.2,0.8]
    for c in countries:
        nationsdict[c].update(trade_volume = tv, trade=tr, nationsdict=nationsdict, capital_mobility = False, partner_develops=pd, dev_shock=dev_shock)
        if t>=500:    
            tv = nationsdict[c].get_trade_volume()
        for v in range(len(variables)):
            for i in industries:
                resultsdict[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])

fig, axs = plt.subplots(4,2, figsize=(15, 20), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)
axs = axs.ravel()

for c in countries:
    for k in range(len(variables)):
        for i in industries:
            axs[k].plot(resultsdict[c][variables[k]][i], label='{}-{}-{}'.format(c, variables[k], i))
            axs[k].legend()
fig.savefig('samuelsonrun.png')

# Setup Gomory-Baumol run with autarky: 0-500 and 1500-2000, Trade:500-1000 and >2000, partner_develops:>1000
resultsdict2 = {}
for c in countries:
    resultsdict2[c] = {}
    for v in variables:
        resultsdict2[c][v]={}
        for i in industries:
            resultsdict2[c][v][i] = []
            
## Create Nations
names = ['USA','CHINA']
count = [500,100]
P_price_wine = [1,1]
P_price_cloth = [1,1]
A_wine = [1,0.2]
A_cloth = [0.5,0.2]
alpha_wine = [0.7, 0.7]
alpha_cloth = [0.4, 0.4]
beta_wine = [0.7, 0.7]
beta_cloth = [0.4, 0.4]
USA_ = Nation(names[0], count[0], P_price_wine[0], P_price_cloth[0], 
              A_wine[0], A_cloth[0], alpha_wine[0], alpha_cloth[0],
              beta_wine[0], beta_cloth[0])
CHINA_ = Nation(names[1], count[1], P_price_wine[1], P_price_cloth[1], 
              A_wine[1], A_cloth[1], alpha_wine[1], alpha_cloth[1],
              beta_wine[1], beta_cloth[1]) 

nationsdict={
    "USA":USA_,
    "CHINA":CHINA_
}


pd=False
tr=False
dev_shock=[0,0]
tv=0
for t in range(3000):  
    if t>500:
        tr = True
    if t>1000:
        pd = True
        dev_shock = [1.5,0.2]
    if t>1500:
        tr=False
    if t>2000:
        tr=True
    for c in countries:
        nationsdict[c].update(trade_volume = tv, trade=tr, nationsdict=nationsdict, capital_mobility = False, partner_develops=pd, dev_shock=dev_shock)
        if t>=500:    
            tv = nationsdict[c].get_trade_volume()
        for v in range(len(variables)):
            for i in industries:
                resultsdict2[c][variables[v]][i].append(getattr(nationsdict[c],functions[v])()[i])

fig, axs = plt.subplots(4,2, figsize=(15, 20), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .5, wspace=.1)
axs = axs.ravel()

for c in countries:
    for k in range(len(variables)):
        for i in industries:
            axs[k].plot(resultsdict2[c][variables[k]][i], label='{}-{}-{}'.format(c, variables[k], i))
            axs[k].legend()
fig.savefig('GBrun.png')



