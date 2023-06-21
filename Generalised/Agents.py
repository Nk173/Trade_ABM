# -*- coding: utf-8 -*-
import math
import random

from functions import demand_function, production_function
from pricing import compute_price_marginal_utilities
import numpy as np
from typing import Dict, List
from wages import wageAsShareOfProduct

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
        self.investment_country = self.nation
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


        p_choice = 0.004
        if capital_mobility:
            # Set initial country to self
            # self.investment_country = self.nation

            # collect the returns from all countries and industries
            r = random.random()
            returns = {}
            for c in countries:
                returns[c] = {}
                for i in industries:
                    returns[c][i] = nationsdict[c].get_ROI()[i]
                    if (self.investment_country==c) and (self.investment_choice==i):
                        self.investment_income = returns[c][i]
            

            if r<p_choice:
                for i in range(len(industries)):
                #     returns[self.nation][industries[i]] = nationsdict[self.nation].get_ROI()[industries[i]]

                    for c in range(len(countries)):
                #         if countries[c] != self.nation:
                #             returns[countries[c]][industries[i]]=  nationsdict[countries[c]].get_ROI()[industries[i]]                        
                        # make the highest 
                        if self.investment_income *1.02 <= returns[countries[c]][industries[i]]:
                            self.investment_income = returns[countries[c]][industries[i]]
                            self.investment_country = countries[c]
                            self.investment_choice = industries[i]
                    
            
            for i in range(len(industries)):
                W[i] = wages[industries[i]]
                if self.job == industries[i]:
                    self.wage = W[i] 

            if r<p_choice:
                j = np.argmax(W)
                if W[j]>= self.wage * 1.02:
                    self.job = industries[j]
                    self.wage = W[j]

            self.income = self.wage + self.investment_income
                

        else: 
            
            if random.random() < p_choice:
                j = np.argmax(W)
                if W[j]>= self.wage * 1.02:
                    self.job = industries[j]
                    self.wage = W[j]
                r = np.argmax(R)
                if R[r] >= self.investment_income * 1.02:
                    self.investment_choice = industries[r]
                    self.investment_income = R[r]

            self.income = self.wage + self.investment_income 


# Define the nation agent
    '''
    Nation has aggregate labor, capital, demand, production. Wages are determined by the marginal returns to output, prices are change gradually based on demand and supply, and return on capital is given by the differences in revenue and labor costs of produce per unit capital.
    
    Under trade, supply changes due to volume of trade which is set based on prices and production of goods in both countries.
    '''
    # Initialisations
class Nation:
    def __init__(self, name, citizen_count, industries, countries,
                 P, A, alpha, beta,
                 pricing_algorithm= compute_price_marginal_utilities, 
                 utility_algorithm='geomtric',
                 wage_algorithm=wageAsShareOfProduct):
        import numpy as np
        self.name = name
        self.other_variables = {}
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
        self.old_supply = {}
        self.wage = {}
        self.wage_bill = {}
        self.ROI = {}
        self.demand = {}
        self.old_demand = {}
        self.prices = {}
        self.old_prices = {}
        self.traded = {} 
        self.trade_volume={}
        self.UT = 1
        self.mrs = {}
        self.pricing_algorithm = pricing_algorithm
        self.utility_algorithm = utility_algorithm
        self.wage_algorithm = wage_algorithm
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
         
    
    def update(self, nationsdict=None, capital_mobility=False):
        self.old_prices = self.prices.copy()
        countries = self.countries
        industries = self.industries
        for c in range(len(countries)):
            if self.name == countries[c]:
                otherNations = [value for key, value in nationsdict.items() if key not in self.name]
        
            # if partner_develops:
            #     if self.name==countries[1]:
            #         self.A = development_shock                       
        
        # Initialise Price vector for self
        P = np.zeros((len(industries)))
        L = np.zeros((len(industries)))
        K = np.zeros((len(industries)))
        self.old_demand = self.demand.copy()
        self.old_supply = self.supply.copy()

        self.old_demand = self.demand.copy()
        self.old_supply = self.supply.copy()
        for i in range(len(industries)):
            self.labor[industries[i]] = 0
            self.capital[industries[i]] = 0
            self.demand[industries[i]] = 0
            P[i] = self.prices[industries[i]]
        
        # No capital mobility
        if ~capital_mobility:
            for citizen in self.citizens:
                citizen.update(nationsdict, capital_mobility = capital_mobility)
                for i in range(len(industries)):
                        if citizen.job == industries[i]:
                            L[i]+=1
                        if citizen.investment_choice == industries[i]:
                            # K[i]+=citizen.investment_income
                            K[i]+=1     
            
                self.demand[industries[i]] = self.demand[industries[i]] + demand_function(citizen.income, P)[i]


        # w/ capital mobility
        else:
            K = np.zeros((len(industries)))

             # calculate local capital
            for citizen in self.citizens:
                citizen.update(nationsdict, capital_mobility = capital_mobility)
                for j in range(len(industries)):
                    if (citizen.investment_choice == industries[j]) and (citizen.investment_country == self.name):
                        #   K[j]+=citizen.investment_income
                        K[j]=K[j]+1   
                    self.demand[industries[j]] = self.demand[industries[j]] + demand_function(citizen.income, P)[j]

            # check for foreign investment
            for otherNation in otherNations:
                for citizen in otherNation.citizens:
                    if citizen.investment_country==self.name:
                        for j in range(len(industries)):
                            if citizen.investment_choice == industries[j]:
                                # K[j]+=citizen.investment_income
                                K[j]=K[j]+1   
        

        for i in range(len(industries)):
            self.labor[industries[i]] = L[i]
            self.capital[industries[i]] = K[i]
            if self.capital[industries[i]]<1:
                self.capital[industries[i]]=1
                
        # Production
        inc_labor = {}
        inc_production = {}
        for i in range(len(industries)):
            self.wage[industries[i]] = 0
            self.wage_bill[industries[i]] = 0
            self.ROI[industries[i]] = 0
            self.supply[industries[i]] = 0
            self.production[industries[i]] = 0
            self.production[industries[i]] = production_function(self.A[i], self.alpha[industries[i]],
                                                                 self.labor[industries[i]], self.beta[industries[i]], 
                                                                 self.capital[industries[i]])
            self.supply[industries[i]] = self.production[industries[i]] 

            # wage, roi = wageAsMarginalProductROIAsResidual(self,i,industries,production_function)
            wage,roi = self.wage_algorithm(self,i,industries, production_function)

            self.wage[industries[i]] = wage
            self.wage_bill[industries[i]] = self.wage[industries[i]] * self.labor[industries[i]]
            self.ROI[industries[i]] = roi          
            
            self.supply[industries[i]] = self.production[industries[i]] 
        



    def updatePricesAndConsume(self,country_export: Dict[str,float], trade = False):
        industries = self.industries
        ## compute the effect of trade...
        if trade == True:
            self.traded = country_export.copy()
            self.resolve_trade(country_export)
            # self.trade_volume = trade_volume[self.nation]
            

        self.pricing_algorithm(self, algorithm = self.utility_algorithm)

    
    def utilityFunction(self, consumption: Dict[str,float], algorithm, weights=None, elasticities=None, sigma=None):
        UT = 1
        if algorithm =='geometric':
            for i in range(len(self.industries)):
                UT = UT * consumption[self.industries[i]]
            # Utility
            UT = UT ** (1 / float(len(self.industries)))

        elif algorithm=='ces':
            w = weights
            s = elasticities
            p = sigma
            weighted_consumption = np.zeros((len(self.industries),1))
            for i in range(len(self.industries)):
                weighted_consumption[i] = w[i] * consumption[self.industries[i]] ** s[i]

            UT = np.sum(weighted_consumption) ** (1 / p)
        return UT

    def resolve_trade(self,exported: Dict[str,float] ):
        ## we receive the total amount traded per good and change the supply
        for i in range(len( self.supply)):
            ## change the supply!
            self.supply[self.industries[i]] -= exported[self.industries[i]]
            ## make sure we didn't fuck up
            #assert self.supply[self.industries[i]]>=0


                 
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
            
            return self.UT
    
    def get_demand(self):
            return self.demand
    
    def get_traded(self):
            return self.traded
        
    def get_supply(self):
            return self.supply
    
    def get_trade_volume(self):
            return self.trade_volume

    def get_MRS(self):
            return self.mrs
    
    def compute_hypothetical_demand(self,P,industries):
        demand = {}
        for i in range(len(industries)):
            demand[industries[i]] = 0
            for citizen in self.citizens:
                demand[industries[i]] += demand_function(citizen.income, P)[i]
        return demand