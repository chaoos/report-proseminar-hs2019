# coding: utf-8

from helpers import *
from bgcurrent import BGCurrent
#from chelpers import *
from testing import Testing, equal
import numpy as np
from timeit import default_timer as timer

start = timer()
n = 30
P = generateMasslessMomenta(n)
q_all = masslessMomentum()
q_plus = masslessMomentum()
q_minus = masslessMomentum()

t = Testing()

f = lambda i: -1 if i==0 or i==1 else 1
h = [f(i) for i in range(n)]

gluons = []
for i in range(n):
    gluons.append(Particle(h[i], P[i], q_all))

gluons = np.array(gluons)
end = timer()
time_ini = end - start
print("time_ini", time_ini)

# manual method
start = timer()
#J__mu = J__a(gluons[:-1]).reshape(1,4)
BG = BGCurrent(gluons[:-1])
J__mu = BG.J__a()
eps_mu = gluons[-1].eps_a
#A_n__tree_my = J__mu.dot(eps_mu)[0,0]
A_n__tree_my = np.einsum('i,i', eps_mu.reshape(4), J__mu.reshape(4))
end = timer()
time_my = end - start
print("time_my", time_my)

# Parke Taylor formula
start = timer()
A_n__tree_PT = PT(gluons)
end = timer()
time_PT = np.nan if np.isnan(A_n__tree_PT) else end - start
print("time_PT", time_PT)

t.test("The two formulas give equal amplitudes (n="+str(n)+")", equal(A_n__tree_my, A_n__tree_PT))
