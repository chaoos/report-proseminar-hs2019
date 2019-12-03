# coding: utf-8

from helpers import *
from testing import Testing, equal, equal3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random
from timeit import default_timer as timer

N = [4, 5, 6, 7, 8, 9, 10]
Y1, Y2, Y3 = [], [], []
Y_PT = []

for n in N:

    P = generateMasslessMomenta(n)

    q_all = masslessMomentum()
    q_plus = masslessMomentum()
    q_minus = masslessMomentum()

    fh = lambda i: -1 if i==0 or i==1 else 1
    fq = lambda i, hi: q_minus if hi == -1 else q_plus
    h = [fh(i) for i in range(n)]
    q = [fq(i, h[i]) for i in range(n)]

    t = Testing()

    gluons = []
    for i in range(n):
        gluons.append(Particle(h[i], P[i], q[i]))

    # recurvive method
    start = timer()
    J__mu = J__a(gluons[:-1], P[:-1])
    eps_mu = gluons[-1].eps_a
    A_n__tree_my1 = J__mu.dot(eps_mu)[0,0]
    end = timer()
    time_my1 = end - start

    # recurvive method with q choice
    start = timer()
    J__mu = J__a_2(gluons[:-1], P[:-1])
    eps_mu = gluons[-1].eps_a
    A_n__tree_my2 = J__mu.dot(eps_mu)[0,0]
    end = timer()
    time_my2 = end - start

    # Parke Taylor formula
    start = timer()
    A_n__tree_PT = PT(gluons)
    end = timer()
    time_PT = np.nan if np.isnan(A_n__tree_PT) else end - start

    t.test("The two formulas give equal amplitudes (n="+str(n)+")", equal3(A_n__tree_my1, A_n__tree_my2, A_n__tree_PT))

    Y1.append(time_my1)
    Y2.append(time_my2)
    Y_PT.append(time_PT)


plt.figure(0)
plt.semilogy(N, Y1, '-', label=r'Recursive')
plt.semilogy(N, Y2, '-', label=r'Recursive with q choice')
plt.semilogy(N, Y_PT, '-', label=r'Parke Taylor')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('plots/comparison3.eps')
plt.show()
