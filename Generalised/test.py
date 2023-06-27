import unittest
# from init import case, countries, count, industries, P, A, alpha, beta
from main import gulden
import numpy as np

class TestGulden(unittest.TestCase):

    def test_exports(self):

        # set-up samuelson run without development
        case = 'test'
        countries = ['USA','CHINA']
        count = [100, 1000]
        industries = ['wine','cloth']

        P={}
        P['USA'] =   [1,1]
        P['CHINA'] = [1,1]

        A={}
        A['USA']=    [0.5, 2]
        A['CHINA'] = [0.2, 0.05]

        alpha={}
        alpha['USA'] =   [0.5, 0.5]
        alpha['CHINA'] = [0.5, 0.5]

        beta={}
        beta['USA'] =   [0.5, 0.5]
        beta['CHINA'] = [0.5, 0.5]
        
        shock={}
        shock['USA'] = [0.5, 2]
        shock['CHINA'] = [0.2, 0.8]

        from pricing import compute_price_marginal_utilities
        from wages import wageAsMarginalProductROIAsResidual
        
        # run model for base case
        production, resultsdict = gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta, total_time = 2000, trade_time = 500,
                                         pd_time=10000, shock=shock, cm_time=6000, autarky_time=5000,
                                         pricing_algorithm =compute_price_marginal_utilities, wage_algorithm = wageAsMarginalProductROIAsResidual, utility_algorithm = 'geometric', plot=False)
        
        diag_diff = production['USA']['cloth']-production['CHINA']['wine']
        if diag_diff<2:
            print('Gulden test complete for base Samuelson Case')
        else:
            print('test failed!')

if __name__=='__main__':
    unittest.main()