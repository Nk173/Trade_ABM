import random
from functions import demand_function, production_function

# Define the citizen agent
class Citizen:
    '''
    Citizen belongs in a nation and has a 1 unit of labor for a wage and an 1 unit of capital for a return. Citizen consumes both goods using   all her income. 
    
    Citizen reconsiders job and invesment choices 0.4% of the time based on the wages and ROI from the 2 industries in the Nation. 
    
    Under capital mobility, citizen can invest in industries abroad if it offers a better return. A fraction (repatriation_pct) of this income can be brought back to the Nation of residence while the remaining can be used to purchase goods in the other country.
    
    '''
    # Initiatilisations
    def __init__(self, nation):
        
        self.nation = nation
        self.income = 0
        self.foreign_income = 0
        self.job = random.choice(['wine', 'cloth'])
        self.investment_choice = random.choice(['wine', 'cloth'])
        self.investment_income = 0
        self.demand = {'wine':0, 'cloth':0}
        self.foreign_demand = {'wine':0, 'cloth':0}
        
    # Updates: income for labor and investment, demand for goods to consume, and considers job and investment choices.
    def update(self, nationsdict, capital_mobility = False):
        if self.nation=='USA':
            otherNation=nationsdict['CHINA']
   
        elif self.nation=='CHINA':
            otherNation=nationsdict['USA']
            
        wages = nationsdict[self.nation].get_wages()
        returns = nationsdict[self.nation].get_ROI()    
        prices = nationsdict[self.nation].get_prices()
        
        Ww = wages['wine']
        Wc = wages['cloth']
        
        Rw = returns['wine']
        Rc = returns['cloth']
        
        Pw = prices['wine']
        Pc = prices['cloth']
        self.investment_income=0
        if self.investment_choice=='wine':
            self.investment_income=Rw
        
        if self.investment_choice=='cloth':
            self.investment_income=Rc
        
        if capital_mobility:
            self.foreign_income=0
            foreign_returns = otherNation.get_ROI()
            Rfw = foreign_returns['wine']
            Rfc = foreign_returns['cloth']

            if self.investment_choice=='fwine':
                self.investment_income = Rfw * repatriation_pct
                self.foreign_income = Rfw* (1-repatriation_pct)

            if self.investment_choice=='fcloth':
                self.investment_income = Rfc * repatriation_pct
                self.foreign_income = Rfc* (1-repatriation_pct)
            
        if random.random() < 0.004:
            if Ww > Wc:
                self.job = 'wine'
            else:
                self.job = 'cloth'

            if Rw > Rc:
                self.investment_choice = 'wine'
                self.investment_income = Rw
            else:
                self.investment_choice = 'cloth'
                self.investment_income = Rc
            
            if capital_mobility:
                self.allow_capital_mobility(otherNation, repatriation_pct)
                            
        self.income = (Ww if self.job == 'wine' else Wc) + self.investment_income 
        
        self.demand = {'wine':0, 'cloth':0}
        self.demand['wine'] = demand_function(self.income, Pw,Pc)[0]
        self.demand['cloth'] = demand_function(self.income, Pw,Pc)[1]
        
        if capital_mobility:
            prices_f = otherNation.get_prices()
            Pfw = prices_f['wine']
            Pfc = prices_f['cloth']
            
            self.foreign_demand = {'wine':0, 'cloth':0}
            self.foreign_demand['wine'] = demand_function(self.foreign_income, Pfw, Pfc)[0]
            self.foreign_demand['cloth'] = demand_function(self.foreign_income, Pfw, Pfc)[1]
        
    
    def allow_capital_mobility(self, otherNation, repatriation_pct):
        self.foreign_income=0

        foreign_returns = otherNation.get_ROI()
        Rfw = foreign_returns['wine']
        Rfc = foreign_returns['cloth']
        
        if self.investment_choice=='fwine':
            self.investment_income = Rfw * repatriation_pct
            self.foreign_income = Rfw * (1-repatriation_pct)
        
        if self.investment_choice=='fcloth':
            self.investment_income = Rfc * repatriation_pct
            self.foreign_income = Rfc * (1-repatriation_pct)
            
        if Rfw>self.investment_income:
                self.investment_choice = 'fwine'
                self.investment_income = Rfw * repatriation_pct
                self.foreign_income = Rfw * (1-repatriation_pct)
                
        if Rfc>self.investment_income:
                self.investment_choice = 'fcloth'
                self.investment_income = Rfc * repatriation_pct
                self.foreign_income = Rfc * (1-repatriation_pct)

                        
# Define the nation agent
class Nation:
    '''
    Nation has aggregate labor, capital, demand, production. Wages are determined by the marginal returns to output, prices are change gradually based on demand and supply, and return on capital is given by the differences in revenue and labor costs of produce per unit capital.
    
    Under trade, supply changes due to volume of trade which is set based on prices and production of goods in both countries.
    '''
    
    # Initialisations
    def __init__(self, name, citizen_count, 
                 P_price_wine=1, P_price_cloth=1,
                 A_wine=0.5, A_cloth=2, alpha_wine=0.5, 
                 alpha_cloth=0.5, beta_wine=0.5, beta_cloth=0.5):
        
        import numpy as np
        self.name = name
        self.citizens = [Citizen(self.name) for _ in range(citizen_count)]
        self.A = {'wine':A_wine, 'cloth':A_cloth}
        self.alpha = {'wine':alpha_wine, 'cloth':alpha_cloth}
        self.beta = {'wine':beta_wine, 'cloth':beta_cloth}
        self.labor = {'wine':0, 'cloth':0}
        self.capital = {'wine':0, 'cloth':0}
        self.production = {'wine': 0, 'cloth': 0}
        self.supply = {'wine': 0, 'cloth': 0}
        self.wage = {'wine': 0, 'cloth': 0}
        self.wage_bill = {'wine':0, 'cloth':0}
        self.ROI = {'wine': 0, 'cloth': 0}
        self.demand = {'wine': 0, 'cloth': 0}
        self.prices = {'wine': P_price_wine, 'cloth': P_price_cloth}
        self.traded = {'wine': 0, 'cloth': 0} 
        self.trade_volume = 0
        
    def update(self, trade_volume, trade = False, nationsdict=None, capital_mobility=False,
               partner_develops=False, dev_shock= [0,0]):
        
        if self.name=='USA':
            otherNation=nationsdict['CHINA']
   
        elif self.name=='CHINA':
            otherNation=nationsdict['USA']
            if partner_develops:
                self.A['wine'] = dev_shock[0]
                self.A['cloth'] = dev_shock[1]
              
        # Labor and Capital
        import numpy as np
        wlab=0
        clab=0
        ccap=0
        wcap=0
        
        self.labor = {'wine':0, 'cloth':0}
        self.capital = {'wine':0, 'cloth':0}
        for citizen in self.citizens:
            citizen.update(nationsdict, capital_mobility=capital_mobility)
            
            if citizen.job=='cloth':
                clab += 1
            else: 
                wlab += 1
                
            if citizen.investment_choice=='cloth':
                ccap += 1
            elif citizen.investment_choice=='wine':
                wcap += 1
            
        self.labor={'wine':wlab, 'cloth':clab}
        self.capital={'wine':wcap, 'cloth':ccap}
            
        if capital_mobility:
            for citizen2 in otherNation.citizens:
                citizen.update(nationsdict, capital_mobility=capital_mobility)
                if citizen2.investment_choice=='fcloth':
                    self.capital['cloth'] += 1
                elif citizen2.investment_choice=='fwine':
                    self.capital['wine'] += 1 
                    
        if self.capital['wine']<1:
            self.capital['wine']=1
        if self.labor['cloth']<1:
            self.labor['cloth']=1
        if self.capital['cloth']<1:
            self.capital['cloth']=1
        if self.labor['wine']<1:
            self.labor['wine']=1
        
        # Domestic Demand
        self.demand = {'wine':0, 'cloth':0}
        for citizen in self.citizens:
            self.demand['wine'] = self.demand['wine'] + citizen.demand['wine']
            self.demand['cloth'] = self.demand['cloth'] + citizen.demand['cloth']
        
        if capital_mobility:
            for citizen in otherNation.citizens:
                self.demand['wine'] = self.demand['wine'] + citizen.foreign_demand['wine']
                self.demand['cloth'] = self.demand['cloth'] + citizen.foreign_demand['cloth']

        # Production
        self.production['wine'] = production_function(self.A['wine'], self.alpha['wine'], self.labor['wine'], self.beta['wine'], self.capital['wine'])
        self.production['cloth'] = production_function(self.A['cloth'], self.alpha['cloth'], self.labor['cloth'], self.beta['cloth'], self.capital['cloth'])

        # Wages
        self.wage = {'wine':0, 'cloth':0}
        inc_labor = 0
        inc_labor = self.labor['wine'] + 1
        inc_production=0
        inc_production = production_function(self.A['wine'], self.alpha['wine'], inc_labor, self.beta['wine'], self.capital['wine'])
        self.wage['wine'] = self.prices['wine'] * (inc_production - self.production['wine'])
        
        inc_labor = 0
        inc_labor = self.labor['cloth'] + 1
        inc_production=0
        inc_production = production_function(self.A['cloth'], self.alpha['cloth'], inc_labor, self.beta['cloth'], self.capital['cloth'])
        self.wage['cloth'] = self.prices['cloth'] * (inc_production - self.production['cloth'])
   
            
        # ROI
        self.wage_bill = {'wine':0, 'cloth':0}
        self.wage_bill['wine'] = self.wage['wine'] * self.labor['wine']
        self.wage_bill['cloth'] = self.wage['cloth'] * self.labor['cloth']
        self.ROI = {'wine':0, 'cloth':0}
        self.ROI['wine'] = ((self.prices['wine'] * self.production['wine']) - self.wage_bill['wine']) / self.capital['wine']
        self.ROI['cloth'] = ((self.prices['cloth'] * self.production['cloth']) - self.wage_bill['cloth']) / self.capital['cloth']
        
        
        # Domestic Supply
        self.supply = {'wine':0, 'cloth':0}
        self.supply['wine'] = self.production['wine'] 
        self.supply['cloth'] = self.production['cloth'] 
        
        if trade==True:
            self.trade_volume = trade_volume
            self.adjust_trade(otherNation)
        
        # Prices (price of wine is set to 1 and is the reference good)
        if self.demand['cloth'] > self.supply['cloth']:
            self.prices['cloth'] = self.prices['cloth'] + (self.prices['cloth']*0.02) 
        else: 
            self.prices['cloth'] = self.prices['cloth'] - (self.prices['cloth']*0.02)
 

       # Utility
        self.national_utility = np.sqrt(self.supply['wine']*self.supply['cloth'])

    # Trade
    def adjust_trade(self, otherNation):
            
            # Check country name
            # if self.A['wine']/self.A['cloth'])>(otherNation.A['wine']/otherNation.A['cloth']):
            self.supply = {'wine':0, 'cloth':0}
            if self.name == 'USA':
                wine0 = self.production['wine']
                wine1 = otherNation.production['wine']
                cloth0 = self.production['cloth']
                cloth1 = otherNation.production['cloth']
                cloth_prices0 = self.prices['cloth']
                cloth_prices1 = otherNation.prices['cloth']
                
            # elif self.A['wine']/self.A['cloth'])>(otherNation.A['wine']/otherNation.A['cloth']):
            elif self.name == 'CHINA':
                wine0 = otherNation.production['wine']
                wine1 = self.production['wine']
                cloth0 = otherNation.production['cloth']
                cloth1 = self.production['cloth']
                cloth_prices0 = otherNation.prices['cloth']
                cloth_prices1 = self.prices['cloth']
                
            if cloth_prices0 > cloth_prices1:
                self.trade_volume = self.trade_volume - 0.5
                
            else:
                self.trade_volume = self.trade_volume + 0.5
                
            if (wine0< self.trade_volume) & (wine1< self.trade_volume):
                if (wine0 > wine1):
                        self.trade_volume = wine0
                else: 
                        self.trade_volume = wine1

            # Supply with Trade enabled
            # If country is USA
            self.traded = {'wine': 0, 'cloth': 0} 
            if self.name == 'USA':
                self.traded['wine'] = self.trade_volume
                self.traded['cloth']= -1*self.trade_volume / self.prices['cloth']
                
            else:
                self.traded['wine'] = -1*self.trade_volume
                self.traded['cloth']= self.trade_volume / self.prices['cloth']  
             
            self.supply['wine'] = self.production['wine'] + self.traded['wine']
            self.supply['cloth']= self.production['cloth'] + self.traded['cloth']
            
            if self.supply['wine']<=0:
                self.supply['wine']= 0.0001
            elif self.supply['cloth']<=0:
                self.supply['cloth']=0.0001
                            
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
        
    def get_supply(self):
            return self.supply
    
    def get_trade_volume(self):
            return self.trade_volume
            