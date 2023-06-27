## Initialisations
# 2x2 case

case = '2x2'

if case == '2x2':
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
    
    shock = [0.2, 0.8]
  

# 2x3 case]
if case == '2x3':
    countries = ['USA','CHINA',]
    count = [100, 100]
    industries = ['wine','cloth','wheat']

    P={}
    P['USA'] = [1,1,1]
    P['CHINA'] = [1,1,1]

    A={}
    A['USA']= [0.5, 2, 1]
    A['CHINA'] = [0.2, 0.05,1]

    alpha={}
    alpha['USA'] = [0.7,0.7, 0.7]
    alpha['CHINA'] = [0.5, 0.5, 0.5]

    beta={}
    beta['USA'] = [0.5, 0.5,0.5]
    beta['CHINA'] = [0.5, 0.5,0.5]

    shock=None

    weights=[1,1] 
    elasticities=[0.5,0.5]
    sigma= 1/0.5
# 3x3 case
if case =='3x3asym':
    countries = ['USA','CHINA', 'INDIA']
    count = [100, 100, 100]
    industries = ['wine','cloth','wheat']

    P={}
    P['USA'] = [1,1,1]
    P['CHINA'] = [1,1,1]
    P['INDIA'] = [1,1,1]

    A={}
    A['USA']= [0.5, 2, 1]
    A['CHINA'] = [0.2, 0.05,1]
    A['INDIA'] = [0.7, 0.3,1]

    alpha={}
    alpha['USA'] = [0.5,0.5, 0.5]
    alpha['CHINA'] = [0.5, 0.5, 0.5]
    alpha['INDIA'] = [0.5, 0.5,0.5]

    beta={}
    beta['USA'] = [0.5, 0.5,0.5]
    beta['CHINA'] = [0.5, 0.5,0.5]
    beta['INDIA'] = [0.5,0.5,0.5]

# 3x3 case symmetrical

if case =='3x3':
    countries = ['USA', 'CHINA', 'INDIA']
    count = [100, 100, 100]
    industries = ['wine', 'cloth', 'wheat']

    P = {}
    P['USA']   = [1, 1, 1]
    P['CHINA'] = [1, 1, 1]
    P['INDIA'] = [1, 1, 1]

    A = {}
    A['USA'] =   [2, 1, 1]
    A['CHINA'] = [1, 2, 1]
    A['INDIA'] = [1, 1, 2]

    alpha = {}
    alpha['USA']   = [0.5, 0.5, 0.5]
    alpha['CHINA'] = [0.5, 0.5, 0.5]
    alpha['INDIA'] = [0.5, 0.5, 0.5]


    beta = {}
    beta['USA']   = [0.5, 0.5, 0.5]
    beta['CHINA'] = [0.5, 0.5, 0.5]
    beta['INDIA'] = [0.5, 0.5, 0.5]

    weights=[1,1,1] 
    elasticities=[0.3,0.3,0.3]
    sigma= 1/0.3

    shock = {}
    shock['USA'] =   [2,1,1]
    shock['CHINA'] = [1,2,1]
    shock['INDIA'] = [1,1,2]

    dist = [[0,1,2],
            [1,0,1],
            [2,1,0]]

# 5x5 case
if case == '5x5':
    countries = ['USA', 'CHINA', 'INDIA', 'JAPAN', 'GHANA']
    count = [500, 400, 300, 200, 100]
    industries = ['wine', 'cloth', 'wheat', 'computers', 'shoes']

    P = {}
    P['USA'] = [1, 1, 1, 1, 1]
    P['CHINA'] = [1, 1, 1, 1, 1]
    P['INDIA'] = [1, 1, 1, 1, 1]
    P['JAPAN'] = [1, 1, 1, 1, 1]
    P['GHANA'] = [1, 1, 1, 1, 1]

    A = {}
    A['USA'] = [2, 1, 1, 1, 1]
    A['CHINA'] = [1, 2, 1, 1, 1]
    A['INDIA'] = [1, 1, 2, 1, 1]
    A['JAPAN'] = [1, 1, 1, 2, 1]
    A['GHANA'] = [1, 1, 1, 1, 2]

    alpha = {}
    alpha['USA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha['CHINA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha['INDIA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha['JAPAN'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    alpha['GHANA'] = [0.5, 0.5, 0.5, 0.5, 0.5]

    beta = {}
    beta['USA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    beta['CHINA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    beta['INDIA'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    beta['JAPAN'] = [0.5, 0.5, 0.5, 0.5, 0.5]
    beta['GHANA'] = [0.5, 0.5, 0.5, 0.5, 0.5]

    shock = {}
    shock['USA'] =   [2,1,1]
    shock['CHINA'] = [1,2,1]
    shock['INDIA'] = [1,1,2]

    weights=[1,1,1] 
    elasticities=[0.2,0.2,0.2,0.2,0.2]
    sigma= 1/0.2