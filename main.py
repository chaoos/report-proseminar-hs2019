# coding: utf-8

from helpers import *
from testing import testing, equal
from fourGluons import fourGluonScattering
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import itertools
import random

# Generate the momenta
n = 4
p = generateMasslessMomenta(n)

# The helicity's
h = [-1, -1, 1, 1]

# create the reference vector (the same for all particles)
q = masslessMomentum()

testing(n, p, h, q)

# create n gluons as Particle objects with the same reference vectors
gluons = []
for i in range(n):
    hi = h[i] if i < len(h) else +1
    gluons.append(Particle(hi, p[i], q))

M, M1, M2, M3, M1_2, M1_3, M3_2 = fourGluonScattering(gluons)
same = [M, M1, M2, M3, M1_2, M1_3, M3_2]

# create n gluons as Particle objects with all different reference vectors
gluons = []
for i in range(n):
    hi = h[i] if i < len(h) else +1
    gluons.append(Particle(hi, p[i]))

M, M1, M2, M3, M1_2, M1_3, M3_2 = fourGluonScattering(gluons)
diff = [M, M1, M2, M3, M1_2, M1_3, M3_2]

test("M",equal(same[0], diff[0]))
test("|M|^2",equal(np.absolute(same[0])**2, np.absolute(diff[0])**2))
test("M1",equal(same[1], diff[1]))
test("M2",equal(same[2], diff[2]))
test("M3",equal(same[3], diff[3]))
test("M1_2",equal(same[4], diff[4]))
test("M1_3",equal(same[5], diff[5]))
test("M3_2",equal(same[6], diff[6]))

test("2 ways give the same result for s-type", [M1 == M1_2, M1, M1_2])
test("2 ways give the same result for 4-gluon vertex", equal(M3, M3_2))

M = 1j*(M1 + M2 + M3)
test("Own implementation: 0 < |M|^2 < 1", [np.absolute(M)**2 < 1 and np.absolute(M)**2 > 0, M])

M_PT = PT(gluons)
test("Parker Taylor: 0 < |M|^2 < 1", [np.absolute(M_PT)**2 < 1 and np.absolute(M_PT)**2 > 0, M_PT])
test("If both formulas give equal amplitudes for 4 gluons", equal(np.absolute(M)**2, np.absolute(M_PT)**2))