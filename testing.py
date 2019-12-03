from helpers import *
import numpy as np
import itertools
import random

class Testing:
    #
    # @param Testing self       reference to the class instance
    # @return Testing           class instance
    #
    def __init__ (self):
        self.summary = []
        self.retval = True

    def test(self, message, condition):
        # A simple test print function to give colored guidlines
        #
        # @param string message The test message
        # @param bool condition pass/fail True/False
        # @return bool pass/fail
        #
        print('{4}[TEST]{3} {5}{0}{3} ? {1}{2}{3}'.format(
            message,
            bcolors.OKGREEN if condition[0] else bcolors.FAIL,
            condition if condition[0] else condition,
            bcolors.ENDC,
            bcolors.OKBLUE,
            bcolors.BOLD
        ))
        try:
            c = condition[0]
        except IndexError:
            c = condition
        self.summary.append([message, condition])
        self.retval = c if c == False else self.retval
        return c



def testfunc(p, h, q):
    """
    Run all tests with the given setup

    # @param array p       the 4-momenta
    # @param array|int h   the helicities corresponding to the 4-momenta
    # @param array q       the reference vectors corresponding to the 4-momenta
    # @return void

    >>> n = 4
    >>> P = generateMasslessMomenta(n)
    >>> h = [-1, -1, 1, 1]
    >>> q = [masslessMomentum() for i in range(n)]
    >>> testfunc(P, h, q) # doctest: +ELLIPSIS
    \033[94m[TEST]\033[0m ...
    ...
    True
    """

    t = Testing()

    n = len(p)
    p_tot = np.sum(p, axis=0)
    r = []

    # tests
    for i in range(n):
        t.test("The 4-vector (p"+str(i)+"^2 = 0)", equal(minkowskiDot(p[i], p[i]), 0))

    for i in range(4):
        t.test("The 4-vectors (Σ_i p_i,"+str(i)+" = 0)", equal(p_tot[i], 0))

    # create n gluons as Particle objects
    gluons = []
    for i in range(n):
        hi = h[i] if i < len(h) else +1
        gluons.append(Particle(hi, p[i], q[i]))

    # det p_αβ
    for i in range(n):
        t.test("det(p"+str(i)+"_αβ)=0", equal(np.linalg.det(gluons[i].p_ab), 0))


    for i in range(n):
        j = i+1 if i < len(gluons)-1 else 0
        g1, g2 = gluons[i], gluons[j]
        lhs = spinor_prod(g1, g2)*spinor_prod2(g2, g1)
        rhs = 2*minkowskiDot(g1.p, g2.p)
        t.test("The spinor product <p"+str(i)+",p"+str(j)+">[p"+str(j)+",p"+str(i)+"] = 2*p"+str(i)+"*p"+str(j)+" ", equal(rhs, lhs))

    # eps_p+ * p
    for i in range(n):
        t.test("ε_p"+str(i)+"+ * p"+str(i)+" = 0", equal(minkowskiDot(gluons[i].eps_plus__a.reshape(4), p[i]), 0))

    # eps_p+ * p
    for i in range(n):
        t.test("ε_p"+str(i)+"- * p"+str(i)+" = 0", equal(minkowskiDot(gluons[i].eps_minus__a.reshape(4), p[i]), 0))

    # eps_p+ * eps_p-
    for i in range(n):
        t.test("ε_p"+str(i)+"+^α * ε_p"+str(i)+"-_α = -1", equal(gluons[i].eps_plus__a.dot(gluons[i].eps_minus_a)[0,0], -1.0))

    # eps_p- * eps_p+
    for i in range(n):
        t.test("ε_p"+str(i)+"-^α * ε_p"+str(i)+"+_α = -1", equal(gluons[i].eps_minus__a.dot(gluons[i].eps_plus_a)[0,0], -1.0))

    # eps_p+ * eps_p+
    for i in range(n):
        t.test("ε_p"+str(i)+"+^α * ε_p"+str(i)+"+_α = 0", equal(gluons[i].eps_plus__a.dot(gluons[i].eps_plus_a)[0,0], 0))

    # eps_p- * eps_p-
    for i in range(n):
        t.test("ε_p"+str(i)+"-^α * ε_p"+str(i)+"-_α = 0", equal(gluons[i].eps_minus__a.dot(gluons[i].eps_minus_a)[0,0], 0))

    # ε_p+-^α * ε_q-+α = 0
    i, j = random.randint(0,n-1), random.randint(0,n-1)
    t.test("ε_p"+str(i)+"+^α * ε_p"+str(j)+"+α = 0", equal(gluons[i].eps_plus__a.dot(gluons[j].eps_plus_a)[0,0], 0))
    t.test("ε_p"+str(i)+"-^α * ε_p"+str(j)+"-α = 0", equal(gluons[i].eps_minus__a.dot(gluons[j].eps_minus_a)[0,0], 0))

    # λ_-p = -λ_p (A.3)
    for i in range(n):
        g_minus_p = Particle(gluons[i].helicity, -gluons[i].p, gluons[i].q)
        t.test("λ_-p"+str(i)+" = -λ_p"+str(i)+"", equal(g_minus_p.λ_pa, -gluons[i].λ_pa))

    # ~λ_-p = ~λ_p (A.3)
    for i in range(n):
        g_minus_p = Particle(gluons[i].helicity, -gluons[i].p, gluons[i].q)
        t.test("~λ_-p"+str(i)+" = ~λ_p"+str(i)+"", equal(g_minus_p.λ_tilda_pa_dot, gluons[i].λ_tilda_pa_dot))

    μ, σ, λ = random.randint(0,3), random.randint(0,3), random.randint(0,3)
    i, j, k = random.randint(0,n-1), random.randint(0,n-1), random.randint(0,n-1)
    pi, pj, pk = p[i], p[j], p[k]
    t.test("V3g(i,j,k) is antisymmetic w.r.t. perm. of any pair of gluons (1 <-> 2)",
     equal(V3g(pi, pj, pk, μ, σ, λ), -V3g(pj, pi, pk, σ, μ, λ)))
    t.test("V3g(i,j,k) is antisymmetic w.r.t. perm. of any pair of gluons (2 <-> 3)",
     equal(V3g(pi, pj, pk, μ, σ, λ), -V3g(pi, pk, pj, μ, λ, σ)))
    t.test("V3g(i,j,k) is antisymmetic w.r.t. perm. of any pair of gluons (1 <-> 3)",
     equal(V3g(pi, pj, pk, μ, σ, λ), -V3g(pk, pj, pi, λ, σ, μ)))

    return t.retval


def equal(a, b, rtol=1e-05, atol=1e-08):
    # Checks if two numbers are approximately equal according to the tolerance
    #
    # @param complex a  1st number
    # @param complex b  2nd number
    # @return bool 
    #
    return np.allclose(a, b, rtol, atol), a, b

def equal3(a, b, c, rtol=1e-05, atol=1e-08):
    # Checks if 3 numbers are approximately equal according to the tolerance
    #
    # @param complex a  1st number
    # @param complex b  2nd number
    # @param complex 3  3rd number
    # @return bool 
    #
    return np.allclose(a, b, rtol, atol) and np.allclose(c, b, rtol, atol) and np.allclose(a, c, rtol, atol), a, b, c

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

if __name__ == "__main__":
    import doctest
    doctest.testmod()

