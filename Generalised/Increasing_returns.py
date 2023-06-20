from main import gulden
from init import countries, count, industries, P, A

## Reinitialising 
case = '3x3_DR'

alpha = {}
alpha['USA'] = [0.5, 0.5, 0.5]
alpha['CHINA'] = [0.5, 0.5, 0.5]
alpha['INDIA'] = [0.5, 0.5, 0.5]


beta = {}
beta['USA'] = [0.5, 0.5, 0.5]
beta['CHINA'] = [0.5, 0.7, 0.5]
beta['INDIA'] = [0.5, 0.5, 0.7]

## Model--Run
gulden(case=case, countries=countries, count=count, industries=industries, P=P, A=A, alpha=alpha, beta=beta)