from pricing import compute_price_immediate_marginal_utility, compute_price_marginal_utilities
from pricing import gd_pricing, demand_gap_pricing
from wages import wagesAsShareOfMarginalProduct, wageAsMarginalProductROIAsResidual, wageAsShareOfProduct
from main import gulden

# paper plots
# ## Samuelson Set-up
# countries = ['USA','CHINA']
# count = [100, 1000]
# industries = ['wine','cloth']

# P={}
# P['USA'] =   [1,1]
# P['CHINA'] = [1,1]

# A={}
# A['USA']=    [0.5, 2]
# A['CHINA'] = [0.2, 0.05]

# alpha={}
# alpha['USA'] =   [0.5, 0.5]
# alpha['CHINA'] = [0.5, 0.5]

# beta={}
# beta['USA'] =   [0.5, 0.5]
# beta['CHINA'] = [0.5, 0.5]
    
# shock = [0.2, 0.8]

# production, resultsdict = gulden(case='samuelson_model_mrs', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
#                                  total_time = 3000, trade_time = 500, pd_time=1000, shock=shock, cm_time=6000, autarky_time=5000,
#                                  pricing_algorithm =compute_price_marginal_utilities, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.0)


# ## Gomory and Baumol's Retainable Industries (increasing and decreasing returns to scale)
# countries = ['USA','GHANA']
# count = [500, 100]
# industries = ['wine','cloth']

# P={}
# P['USA'] =   [1,1]
# P['GHANA'] = [1,1]

# A={}
# A['USA']=    [1, .5]
# A['GHANA'] = [0.2, 0.2]

# alpha={}
# alpha['USA'] =   [0.7, 0.4]
# alpha['GHANA'] = [0.7, 0.4]

# beta={}
# beta['USA'] =   [0.7, 0.4]
# beta['GHANA'] = [0.7, 0.4]
    
# shock = [1.5, 0.2]

# production, resultsdict = gulden(case='GB_model', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
#                                  total_time = 3000, trade_time = [500,2000], pd_time=1000, shock=shock, cm_time=6000, autarky_time=1500,
#                                  pricing_algorithm =demand_gap_pricing, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.0)

# ## Gomory and Baumol's Retainable Industries (increasing and decreasing returns to scale) using MRS smoothing of prices
# countries = ['USA','GHANA']
# count = [500, 100]
# industries = ['wine','cloth']

# P={}
# P['USA'] =   [1,1]
# P['GHANA'] = [1,1]

# A={}
# A['USA']=    [1, .5]
# A['GHANA'] = [0.2, 0.2]

# alpha={}
# alpha['USA'] =   [0.7, 0.4]
# alpha['GHANA'] = [0.7, 0.4]

# beta={}
# beta['USA'] =   [0.7, 0.4]
# beta['GHANA'] = [0.7, 0.4]
    
# shock = [1.5, 0.2]

# production, resultsdict = gulden(case='GB_model_mrs', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
#                                  total_time = 3000, trade_time = [500,2000], pd_time=1000, shock=shock, cm_time=6000, autarky_time=1500,
#                                  pricing_algorithm =compute_price_marginal_utilities, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.0)

## Gomory and Baumol's Retainable Industries (increasing and decreasing returns to scale) with Capital Mobility
countries = ['USA','GHANA']
count = [500, 100]
industries = ['wine','cloth']

P={}
P['USA'] =   [1,1]
P['GHANA'] = [1,1]

A={}
A['USA']=    [1, .5]
A['GHANA'] = [0.2, 0.2]

alpha={}
alpha['USA'] =   [0.7, 0.4]
alpha['GHANA'] = [0.7, 0.4]

beta={}
beta['USA'] =   [0.7, 0.4]
beta['GHANA'] = [0.7, 0.4]
    
shock = [1.5, 0.2]

production, resultsdict = gulden(case='capmob_model_dgp', countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, 
                                 total_time = 3000, trade_time = [500,2000], pd_time=1000, shock=shock, cm_time=2500, autarky_time=1500,
                                 pricing_algorithm =demand_gap_pricing, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', d=0.0)





