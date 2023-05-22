# -*- coding: utf-8 -*-
import math
import random

from pricing import compute_price_marginal_utilities
from functions import demand_function, production_function
import numpy as np
from typing import Dict, List

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
            if W[j]>= self.wage * 1.05:
                self.job = industries[j]
                self.wage = W[j]
            r = np.argmax(R)
            if R[r] >= self.investment_income * 1.05:
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
                 pricing_algorithm= compute_price_marginal_utilities):
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
        self.old_prices = {}
        self.traded = {} 
        self.trade_volume={}
        self.UT = 1
        self.mrs = {}
        self.pricing_algorithm = pricing_algorithm
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
         
    
    def update(self, nationsdict=None):
        self.old_prices = self.prices.copy()
        countries = self.countries
        industries = self.industries
        for c in range(len(countries)):
            if self.name == countries[c]:
                otherNation = [value for key, value in nationsdict.items() if key not in self.name]
            # if partner_develops:
            #     if self.name==countries[1]:
            #         self.A = development_shock
                    
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
        



    def updatePricesAndConsume(self,country_export: Dict[str,float], trade = False):
        industries = self.industries
        ## compute the effect of trade...
        if trade == True:
            self.traded = country_export.copy()
            self.resolve_trade(country_export)
            # self.trade_volume = trade_volume[self.nation]

        self.pricing_algorithm(self)

    def utilityFunction(self, consumption: Dict[str,float], algorithm = 'geometric'):
        UT = 1
        if algorithm =='geometric':
            for i in range(len(self.industries)):
                UT = UT * consumption[self.industries[i]]
            # Utility
            UT = UT ** (1 / float(len(self.industries)))
        
        # elif algorithm=='generalised cobbdouglas':
        #     for i in range(len(self.industries)):
                
                  
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