import random
from functions import demand_function, production_function, RCA
import numpy as np
# Define the citizen agent
class Citizen:
    '''
    Citizen belongs in a nation and has a 1 unit of labor for a wage and an 1 unit of capital for a return. Citizen consumes both goods using   all her income. 
    
    Citizen reconsiders job and invesment choices 0.4% of the time based on the wages and ROI from the 2 industries in the Nation. 
    
    Under capital mobility, citizen can invest in industries abroad if it offers a better return. A fraction (repatriation_pct) of this income can be brought back to the Nation of residence while the remaining can be used to purchase goods in the other country.
    
    '''
    # Initiatilisations
    def __init__(self, nation, industries, countries):
        self.nation = nation
        self.income = 0
        self.wage = 0 
        self.foreign_income = 0
        self.job = random.choice(industries)
        self.investment_choice = random.choice(industries)
        self.investment_income = 0
        self.industries = industries
        self.countries=countries
    
    def update(self, nationsdict, capital_mobility = False):
        industries = self.industries
        countries = self.countries
        for c in range(len(countries)):
            if self.nation == countries[c]:
                otherNation = [key for key, value in nationsdict.items() if key not in self.nation]

        wages = nationsdict[self.nation].get_wages()
        returns = nationsdict[self.nation].get_ROI()    
        prices = nationsdict[self.nation].get_prices()
        W = np.zeros((len(industries)))
        R = np.zeros((len(industries)))
        P = np.zeros((len(industries)))

        for i in range(len(industries)):
            W[i] = wages[industries[i]]
            R[i] = returns[industries[i]]
            P[i] = prices[industries[i]]

            if self.investment_choice == industries[i]:
                self.investment_income = R[i]
                
            if self.job == industries[i]:
                self.wage = W[i]
        
        if random.random() < 0.004:
            j = np.argmax(W)
            self.job = industries[j]
            self.wage = W[j]
            r = np.argmax(R)
            self.investment_choice = industries[r]
            self.investment_income = R[r]

        self.income = self.wage + self.investment_income 


# Define the nation agent
class Nation:
    '''
    Nation has aggregate labor, capital, demand, production. Wages are determined by the marginal returns to output, prices are change gradually based on demand and supply, and return on capital is given by the differences in revenue and labor costs of produce per unit capital.
    
    Under trade, supply changes due to volume of trade which is set based on prices and production of goods in both countries.
    '''
    
    # Initialisations
class Nation:
    def __init__(self, name, citizen_count, industries, countries,
                 P, A, alpha, beta):
        import numpy as np
        self.name = name
        self.industries = industries
        self.countries = countries
        self.citizens = [Citizen(self.name, industries, countries) for _ in range(citizen_count)]
        self.A = A
        self.alpha = {}
        self.beta = {}
        self.labor = {}
        self.capital = {}
        self.production = {}
        self.supply = {}
        self.wage = {}
        self.wage_bill = {}
        self.ROI = {}
        self.demand = {}
        self.prices = {}
        self.traded = {} 
        self.trade_volume={}
        for c in countries:
            self.trade_volume[c] = {}
            for n in countries:
                self.trade_volume[c][n]=0
        for i in range(len(industries)):
            self.alpha[industries[i]] = alpha[i]
            self.beta[industries[i]] = beta[i]
            self.labor[industries[i]] = 0
            self.capital[industries[i]] = 0
            self.production[industries[i]] = 0
            self.supply[industries[i]] = 0
            self.wage[industries[i]] = 0
            self.wage_bill[industries[i]] = 0
            self.ROI[industries[i]] = 0
            self.demand[industries[i]] = 0
            self.prices[industries[i]] = P[i]
            self.traded[industries[i]] = 0 
         
    
    def update(self, trade_volume, trade = False, nationsdict=None, capital_mobility=False, partner_develops=False):
        countries = self.countries
        industries = self.industries
        for c in range(len(countries)):
            if self.name == countries[c]:
                otherNation = [value for key, value in nationsdict.items() if key not in self.name]
            if partner_develops:
                if self.name==countries[1]:
                    self.A = development_shock
                    
        # print(otherNation)
        
        
        # Labor and Capital
        L = np.zeros((len(industries)))
        K = np.zeros((len(industries)))
        P = np.zeros((len(industries)))

        for i in range(len(industries)):
            self.labor[industries[i]] = 0
            self.capital[industries[i]] = 0
            self.demand[industries[i]] = 0
            P[i] = self.prices[industries[i]]
        
        for citizen in self.citizens:
            citizen.update(nationsdict, capital_mobility = False)
            for i in range(len(industries)):
                if citizen.job == industries[i]:
                    L[i]+=1
                if citizen.investment_choice == industries[i]:
                    K[i]+=1      
                self.demand[industries[i]] = self.demand[industries[i]] + demand_function(citizen.income, P)[i]
                    
        for i in range(len(industries)):
            self.labor[industries[i]] = L[i]
            self.capital[industries[i]] = K[i]
            if self.capital[industries[i]]<1:
                self.capital[industries[i]]=1

        # Production
        inc_labor = {}
        inc_production = {}
        UT = 1
        for i in range(len(industries)):
            self.wage[industries[i]] = 0
            self.wage_bill[industries[i]] = 0
            self.ROI[industries[i]] = 0
            self.supply[industries[i]] = 0
            self.production[industries[i]] = 0
            self.production[industries[i]] = production_function(self.A[i], self.alpha[industries[i]], 
                                                                 self.labor[industries[i]], self.beta[industries[i]], 
                                                                 self.capital[industries[i]])
            inc_labor[industries[i]] = 0
            inc_labor[industries[i]] = self.labor[industries[i]] + 1
                     
            inc_production[industries[i]] = production_function(self.A[i], self.alpha[industries[i]], 
                                             inc_labor[industries[i]], self.beta[industries[i]], self.capital[industries[i]])
            
            self.wage[industries[i]] = self.prices[industries[i]] * (inc_production[industries[i]] - self.production[industries[i]])
            # self.wage[industries[i]] = (self.prices[industries[i]] * self.production[industries[i]])/self.labor[industries[i]]
            self.wage_bill[industries[i]] = self.wage[industries[i]] * self.labor[industries[i]]
            
            self.ROI[industries[i]] = ((self.prices[industries[i]] * self.production[industries[i]]) - self.wage_bill[industries[i]]) / self.capital[industries[i]]
            
            self.supply[industries[i]] = self.production[industries[i]] 
        
        
        if trade==True:
            # self.trade_volume = trade_volume[self.nation]
            self.adjust_trade(otherNation, trade_volume)
                     
        # Prices (price of wine is set to 1 and is the reference good)
        for i in range(1,len(industries)):
            if self.demand[industries[i]] > self.supply[industries[i]]:
                self.prices[industries[i]] = self.prices[industries[i]] + (self.prices[industries[i]]*0.02) 
            else: 
                self.prices[industries[i]] = self.prices[industries[i]] - (self.prices[industries[i]]*0.02)   
 
        for i in range(len(industries)):
            UT *= self.supply[industries[i]]
        
        
        # Utility
        self.national_utility = np.sqrt(UT)
        
    def adjust_trade(self, otherNation, trade_volume): 
        countries = self.countries
        industries = self.industries
        trade_volume = trade_volume
        # Initialise supply and trade
        self.supply = {}
        self.traded = {}
        
        for i in industries:
            self.supply[i] = self.production[i]
            self.traded[i] = 0
        
        # For each trade partner
        import random
        otherNation = random.sample(otherNation, len(otherNation))
        
        for trade_partner in otherNation:
            R = RCA(self.A, trade_partner.A)
            
            # Check prices of Non-reference goods
            Q1 = np.zeros((len(industries)))
            Q0= np.zeros((len(industries)))
            p0= np.zeros((len(industries)))
            p1= np.zeros((len(industries)))
                   
            for i in range(len(industries)):
                
                if R[i] == 1:
                    Q0[i] = self.supply[industries[i]]
                    p0[i] = self.prices[industries[i]]
                    Q1[i] = trade_partner.supply[industries[i]]
                    p1[i] = trade_partner.prices[industries[i]]

                else: 
                    Q1[i] = self.supply[industries[i]]
                    p1[i] = self.prices[industries[i]]
                    Q0[i] = trade_partner.supply[industries[i]]
                    p0[i] = trade_partner.prices[industries[i]]
                    
                if i > 0:
                    if p0[i] > p1[i]:
                        trade_volume[self.name][trade_partner.name] = trade_volume[self.name][trade_partner.name] - 0.5

                    else:
                        trade_volume[self.name][trade_partner.name] = trade_volume[self.name][trade_partner.name] + 0.5


                    if (Q0[0] < trade_volume[self.name][trade_partner.name]) & (Q1[0]< trade_volume[self.name][trade_partner.name]):
                        if (Q0[0] > Q1[0]):
                                trade_volume[self.name][trade_partner.name] = Q0[0]
                        else: 
                                trade_volume[self.name][trade_partner.name] = Q1[0]

                    if R[i]==1:
                        # check if you have production to export
                            self.traded[industries[i]] =  self.traded[industries[i]]+ (-1*trade_volume[self.name][trade_partner.name]/self.prices[industries[i]])
                            self.traded[industries[0]] =  self.traded[industries[0]]+(trade_volume[self.name][trade_partner.name]/self.prices[industries[0]])
                    else:
                            self.traded[industries[i]] = self.traded[industries[i]]+ (trade_volume[self.name][trade_partner.name] / trade_partner.prices[industries[i]])
                            self.traded[industries[0]] = self.traded[industries[0]]+(-1*trade_volume[self.name][trade_partner.name]/self.prices[industries[0]])

                    self.supply[industries[i]] = self.supply[industries[i]] + self.traded[industries[i]]
                    self.supply[industries[0]] = self.supply[industries[0]] + self.traded[industries[0]]
                
                if self.supply[industries[i]]<=0:
                        self.supply[industries[i]]= 0.0001

            trade_volume[trade_partner.name][self.name] = trade_volume[self.name][trade_partner.name]
            self.trade_volume = trade_volume
        
        
                 
    def get_capital(self):
            return self.capital        
            
    def get_prices(self):
            return self.prices
        
    def get_labor(self):
            return self.labor
    
    def get_production(self):
            return self.production
        
    def get_wages(self):
            return self.wage
        
    def get_ROI(self):
            return self.ROI
    
    def get_utility(self):
            return self.national_utility
    
    def get_demand(self):
            return self.demand
    
    def get_traded(self):
            return self.traded
        
    def get_supply(self):
            return self.supply
    
    def get_trade_volume(self):
            return self.trade_volume