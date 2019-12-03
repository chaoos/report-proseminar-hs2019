# coding: utf-8

from helpers import *
from testing import Testing, equal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random
from timeit import default_timer as timer

N = [2, 4, 5, 6, 7]
Y1, Y2 = [], []
Y = dict()

for n in N:

    P = generateMasslessMomenta(n)

    q_all = masslessMomentum()
    q_plus = masslessMomentum()
    q_minus = masslessMomentum()

    q_configs = {
        0: {
            'desc': r'all q equal',
            'f': lambda i, hi: q_all,
        },
        1: {
            'desc': r'all q different',
            'f': lambda i, hi: masslessMomentum(),
        },
        2: {
            'desc': r'same q for pos/neg h respectively',
            'f': lambda i, hi: q_minus if hi == -1 else q_plus,
        },
    }

    h_configs = {
        0: {
            'desc': r'first two h -1, rest +1',
            'f': lambda i: -1 if i==0 or i==1 else 1,
        },
    }
    """
        1: {
            'desc': r'all h positive',
            'f': lambda i: +1,
        },
        2: {
            'desc': r'all h negative',
            'f': lambda i: +1,
        },
        3: {
            'desc': r'1 h negative',
            'f': lambda i: -1 if i==0 else 1,
        },
        4: {
            'desc': r'3 h negative',
            'f': lambda i: -1 if i==0 or i==1 or i==2 else 1,
        },
    }
    """

    j = 1
    t = Testing()
    for hi, qi in itertools.product(h_configs, q_configs):

        h = [h_configs[hi]['f'](i) for i in range(n)]
        q = [q_configs[qi]['f'](i, h[i]) for i in range(n)]

        q_desc = q_configs[qi]['desc']

        gluons = []
        for i in range(n):
            gluons.append(Particle(h[i], P[i], q[i]))

        # manual method
        start = timer()
        J__mu = J__a(gluons[:-1])
        print(J__mu)
        eps_mu = gluons[-1].eps_a
        A_n__tree_my = J__mu.dot(eps_mu)[0,0]
        end = timer()
        time_my = end - start

        # Parke Taylor formula
        start = timer()
        A_n__tree_PT = PT(gluons)
        end = timer()
        time_PT = np.nan if np.isnan(A_n__tree_PT) else end - start

        t.test("CASE "+str(j)+": The two formulas give equal amplitudes (n="+str(n)+", "+h_configs[hi]['desc']+", "+q_configs[qi]['desc']+")", equal(A_n__tree_my, A_n__tree_PT))

        #t.test("CASE "+str(j)+":  The two formulas give equal magnitudes ("+h_configs[hi]['desc']+", "+q_configs[qi]['desc']+")", equal(np.abs(A_n__tree_my)**2, np.abs(A_n__tree_PT)**2))

        if Y.get(q_desc, True) == True:
            Y[q_desc] = []

        if Y.get(q_desc+"_PT", True) == True:
            Y[q_desc+"_PT"] = []

        Y[q_desc].append(time_my)
        Y[q_desc+"_PT"].append(time_PT)

        Y1.append(time_my)
        Y2.append(time_PT)

        j = j+1


"""
plt.figure(0)
plt.semilogy(N, Y1, '-', label=r'Own Implementation')
plt.semilogy(N, Y2, '-', label=r'Parke Taylor')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('plots/comparison.eps')
plt.show()
"""


plt.figure(1)
for qi in q_configs:
    q_desc = q_configs[qi]['desc']
    plt.semilogy(N, Y[q_desc], '-', label=r'Recursive Implementation (case '+str(qi+1)+')')

print(N, Y[q_configs[0]['desc']+"_PT"])
plt.semilogy(N, Y[q_configs[0]['desc']+"_PT"], '-', label=r'Parke Taylor')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('plots/comparison2.eps')
plt.show()
