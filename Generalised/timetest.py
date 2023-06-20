from timeit import default_timer as timer
from main import gulden
import numpy as np
import itertools
import random
import string
import matplotlib.pyplot as plt
T = 10
time = []
for i in range(2,T+1):
    countries = random.sample(string.ascii_letters,i+1)
    count = [x*100 for x in range(1,i+1)]
    print(count)
    case = range(i)
    industries = random.sample(string.ascii_letters,i+1)
    alpha = {}
    beta = {}
    P={}
    A={}
    for j in range(i):
        alpha[countries[j]] = [0.5 for x in range(i)]
        beta[countries[j]] = [0.5 for x in range(i)]
        P[countries[j]] = [1 for x in range(i)]
        A[countries[j]] = [1 for x in range(i)]
        A[countries[j]][j] = 2
    print(alpha)

    # print(A[countries[j]][0][j])]

    t1 = timer()
    # print(t1)
    gulden(case=case, countries=countries, count=count, industries=industries,P=P,A=A,alpha=alpha,beta=beta, shock=None)
    t2 = timer()
    time.append(t2-t1)

plt.plot(range(T), time, 'o')
plt.yscale('log')
plt.title('Time Taken for running n by n case gulden model')
plt.savefig('Generalised/plots/timetest.png')







    

    
    # countries, count, industries, P, A, alpha, beta, shock