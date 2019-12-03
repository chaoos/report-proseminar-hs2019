# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.optimize
import numpy.linalg
from random import uniform
import warnings

rand = lambda : np.random.uniform(-1,1)
rand2 = lambda : np.random.uniform(-2,2)
metric = np.diag([1.0, -1.0, -1.0, -1.0])
g = metric

# g_αβ
g_ab = g

#gᵅᵝ
g__ab = g

# Pauli matrices
sigma__0 = np.array([[ 1.0,  0.0 ],
                     [ 0.0,  1.0 ]])
sigma__1 = np.array([[ 0.0,  1.0 ],
                     [ 1.0,  0.0 ]])
sigma__2 = np.array([[ 0.0, -1.0j],
                     [ 1.0j, 0.0 ]])
sigma__3 = np.array([[ 1.0,  0.0 ],
                     [ 0.0, -1.0 ]])
sigma_bar__0 = sigma__0
sigma_bar__1 = -sigma__1
sigma_bar__2 = -sigma__2
sigma_bar__3 = -sigma__3

# Levi Civita tensor with 2 indices: alpha (a), beta (b)
# Indices up
# εᵅᵝ
eps__ab = np.array([[ 0.0, 1.0 ],
                    [-1.0, 0.0 ]])
# Indices down
# ε_αβ
eps_ab = -eps__ab

#
# Generates n 4-momenta with the following boundaries
# * each particle is massless: p^2 = 0 for each 4-momenta p
# * total momentum is zero: p_1 + ... + p_n = 0
#
# @param Problem self   reference to the class instance
# @param integer n      the number of 4-momenta to produce (n >= 2)
# @return array         the n 4-momenta in a 2d array (n x 4 - matrix)
#
def generateMasslessMomenta(n, it = 1):
    if n == 2:
        x = masslessMomentum()
        return np.array([x, -x])

    ret = np.zeros((n, 4), dtype=complex)

    # the first n-2 values, can be randomly rolled
    for x in range(0, n - 2):
        ret[x] = masslessMomentum()

    # p = y + z, where y=p_n-1 and z=p_n, the last two momenta
    p = - np.sum(ret, axis=0)
    y = np.array([0., 0., rand(), rand()], dtype=complex)
    z = np.array([0., 0., p[2] - y[2], p[3] - y[3]], dtype=complex)

    # function to minimize, x=[y_0, y_1, z_0, z_1]
    f = lambda x : np.array([
        x[0]**2 - x[1]**2 - y[2]**2 - y[3]**2,
        x[2]**2 - x[3]**2 - z[2]**2 - z[3]**2,
        x[0] + x[2] - p[0],
        x[1] + x[3] - p[1],
    ])

    # derivative of f (Jacobi matrix)
    df = lambda x : np.array([
        [  2*x[0],  -2*x[1],   0.,      0.       ],
        [  0.,      0.,        2*x[2],  -2*x[3]  ],
        [  1.,      0.,        1.,      0.       ],
        [  0.,      1.,        0.,      1.       ],
    ])

    # initial guess => random values
    x_init = np.array([rand(), rand(), rand(), rand()]) +0j
    maxit = 1000
    x0, iterations = newton(x_init, f, df, maxit = maxit)
    y[0], y[1], z[0], z[1] = list(x0)

    if iterations == maxit:
        #print("it=",it)
        return generateMasslessMomenta(n, it + 1)

    #print("tries=",it)
    #print("iterations=",iterations)

    # append to result array
    ret[n-2] = y # p_{n-1}
    ret[n-1] = z # p_n
    return ret


"""
Newtonapproximation in n-dim (n>=1)
@param callable f         function that takes at least one argument
@param callable Df        Jacobianmatrix of f
@param ndarray|float xk   start values
@param float tol          tolerance (default: 10^-14)
@param int maxit          max iterations (default: 10'000)
@return ndarray|float     approximated solution for f(xk) = 0 within a tolerance
@return int               used iterations
@return ndarray           values of xk in each step
"""
def newton(xk, f, Df, damper=1., tol=10**-14, maxit=10000):

    X = []
    for i in range(1, maxit+1):
        sk = np.linalg.solve(np.atleast_2d(Df(xk)), np.atleast_1d(f(xk)))
        lamk = damper(f, Df, xk, sk) if callable(damper) else damper
        xk -= lamk*sk
        if np.linalg.norm(sk) <= tol*np.linalg.norm(xk):
            break
        X.append(np.array(xk[:]))

    return xk, i #, np.array(X)

#
# Generates a 4-momentum of a massless particle: p^2 = 0
#
# @return array the 4-momentum as an array
#
def masslessMomentum (p1 = None, p2 = None, p3 = None, gen = lambda: rand()):
    p1 = gen() if p1 == None else p1
    p2 = gen() if p2 == None else p2
    p3 = gen() if p3 == None else p3
    sign = np.random.choice([-1, 1])
    p0 = sign*np.linalg.norm(np.array([p1, p2, p3]))
    return np.array([p0, p1, p2, p3], dtype=complex)

# Produces the Minkowski scalar product according to the metric [[metric]].
def minkowskiDot (a, b):
    return g[0][0]*a[0]*b[0] + g[1][1]*a[1]*b[1] + g[2][2]*a[2]*b[2] + g[3][3]*a[3]*b[3]


# Produces the Minkowski scalar product according to the metric [[metric]].
def md (a, b):
    return minkowskiDot(a, b)

#
# class Particle
#
# Convention in function naming:
# * One underline _ denotes a subscript
# * Two underlines __ denote a superscript
# * Greek alpha -> a, beta -> b, ...
# Example: λᵦᵅ --> def λ_b__a():
#
class Particle:
    #
    # @param Particle self      reference to the class instance
    # @param array helicity     the helicity, defaults to +1
    # @param array p            the 4-momentum, defaults to an random massless momentum
    # @param array q            the 4-momentum reference vector
    # @return Particle          class instance
    #
    def __init__ (self,
            # default values
            helicity = +1,
            p = masslessMomentum(),
            q = masslessMomentum()
        ):

        self.p = p
        self.helicity = helicity
        self.q = q

    @property
    def p_plus(self):
        return complex(self.p[0] + self.p[3])

    @property
    def p_minus(self):
        return complex(self.p[0] - self.p[3])

    @property
    def p_orth(self):
        return complex(self.p[1] + 1j*self.p[2])

    @property
    def p_orth_bar(self):
        return complex(self.p[1] - 1j*self.p[2])

    # according to A.2
    # λ_pα = |p>_α -> column vector
    @property
    def λ_pa(self):
        return (+1j)**(heaviside(-self.p[0]))*np.array([
            [-self.p_orth_bar/np.sqrt(self.p_plus)],
            [np.sqrt(self.p_plus)],
        ])

    # according to A.2
    # ~λ_pά = [p|_ά -> row vector
    @property
    def λ_tilda_pa_dot(self):
        return (-1j)**(heaviside(-self.p[0]))*np.array([[
            -self.p_orth/np.sqrt(self.p_plus),
            np.sqrt(self.p_plus),
        ]])

    # λ_p^α = <p|^α -> row vector
    @property
    def λ_p__a(self):
        return np.transpose(eps__ab.dot(self.λ_pa))

    # ~λ_p^ά = |p]^ά -> column vector
    @property
    def λ_tilda_p__a_dot(self):
        return eps__ab.dot(np.transpose(self.λ_tilda_pa_dot))

    # according to 2.7
    # ε_p+^α -> row vector
    @property
    def eps_plus__a(self):
        # null vector
        q = Particle(+1, self.q)
        return (+1/np.sqrt(2))*1/(spinor_prod(q, self))*np.array([[
            q.λ_p__a.dot(sigma__0.dot(self.λ_tilda_p__a_dot))[0,0],
            q.λ_p__a.dot(sigma__1.dot(self.λ_tilda_p__a_dot))[0,0],
            q.λ_p__a.dot(sigma__2.dot(self.λ_tilda_p__a_dot))[0,0],
            q.λ_p__a.dot(sigma__3.dot(self.λ_tilda_p__a_dot))[0,0],
        ]])

    # according to 2.7
    # ε_p-^α -> row vector
    @property
    def eps_minus__a(self):
        # null vector
        q = Particle(+1, self.q)
        return (-1/np.sqrt(2))*1/(spinor_prod2(q, self))*np.array([[
            q.λ_tilda_pa_dot.dot(sigma_bar__0.dot(self.λ_pa))[0,0],
            q.λ_tilda_pa_dot.dot(sigma_bar__1.dot(self.λ_pa))[0,0],
            q.λ_tilda_pa_dot.dot(sigma_bar__2.dot(self.λ_pa))[0,0],
            q.λ_tilda_pa_dot.dot(sigma_bar__3.dot(self.λ_pa))[0,0],
        ]])

    # ε_p+_α -> column vector
    @property
    def eps_plus_a(self):
        # 1st reshape: makes a row vector from ε_p-^α
        # 2nd reshape: makes a row vector as result
        return g_ab.dot(self.eps_plus__a.reshape(4,1)).reshape(4,1)

    # ε_p-_α -> column vector
    @property
    def eps_minus_a(self):
        return g_ab.dot(self.eps_minus__a.reshape(4,1)).reshape(4,1)

    # p_αβ
    # @return 2x2 matrix representation of p with det=0 <=> p^2=0
    @property
    def p_ab(self):
        return self.λ_pa.dot(self.λ_tilda_pa_dot)

    # p^αβ
    # @return a 2x2 matrix with det=0 <=> p^2=0
    @property
    def p__ab(self):
        return self.λ_tilda_p__a_dot.dot(self.λ_p__a)

    # according to 2.7
    # ε_p+
    @property
    def eps__a(self):
        if (self.helicity == +1):
            return self.eps_plus__a
        elif (self.helicity == -1):
            return self.eps_minus__a
        else:
            return None

    # according to 2.7
    # ε_p+
    @property
    def eps_a(self):
        if (self.helicity == +1):
            return self.eps_plus_a
        elif (self.helicity == -1):
            return self.eps_minus_a
        else:
            return None

    # according to 2.7
    # ε_p+
    @property
    def ε__a(self):
        return self.eps__a

    # according to 2.7
    # ε_p+
    @property
    def ε_a(self):
        return self.eps_a

# Feynman rule for the gluon propagator
#
# @param array p    4-momentum of the virtual particle
# @param int μ      Lorentz index initial (can have values 0,1,2,3)
# @param int ν      Lorentz index final   (can have values 0,1,2,3)
# @return complex
#
#     p
# μ ----- ν
#
def Pg(p, μ, ν):
    return -1j*g_ab[μ][ν]/minkowskiDot(p, p)

# Feynman rule for the 3-gluon vertex
#
# @param array k    4-momentum of gluon 1
# @param array p    4-momentum of gluon 2
# @param array q    4-momentum of gluon 3
# @param int λ      Lorentz index of gluon 1 (can have values 0,1,2,3)
# @param int μ      Lorentz index of gluon 2 (can have values 0,1,2,3)
# @param int ν      Lorentz index of gluon 3 (can have values 0,1,2,3)
# @return complex
#
#    p,μ
#     |
#    / \
# k,λ   q,ν
#
def V3g(k, p, q, λ, μ, ν):
    return (1j/np.sqrt(2))*(g__ab[λ][μ]*(k-p)[ν] + g__ab[μ][ν]*(p-q)[λ] + g__ab[ν][λ]*(q-k)[μ])

# Feynman rule for the 4-gluon vertex
#
# @param int λ  Lorentz index of gluon 1
# @param int μ  Lorentz index of gluon 2
# @param int ν  Lorentz index of gluon 3
# @param int ν  Lorentz index of gluon 4
# @return complex
#
#  μ    ν
#    \/
#    /\
#  λ    ρ
#
def V4g(λ, μ, ν, ρ):
    return 1j*g__ab[λ][ν]*g__ab[μ][ρ] - 0.5j*(g__ab[λ][μ]*g__ab[ν][ρ] + g__ab[λ][ρ]*g__ab[μ][ν])

# Heaviside step function
# @param float x
# @param float x0 value to return if x=0
# @return float
def heaviside(x, x0 = 0.0):
    if (x > 0):
        return 1.0
    elif (x < 0):
        return 0.0
    else:
        return x0


# Spinor product <pq>
# <pq> = λ_p^α * λ_qα
#
# @param Particle p   Particle instance of the 1st particle
# @param Particle q   Particle instance of the 2nd particle
# @return complex     the spinor product <pq>
def spinor_prod(p, q):
    return p.λ_p__a.dot(q.λ_pa)[0][0]


# Spinor product [pq]
# [pq] = ~λ_pά * ~λ_q^ά
#
# @param Particle p   Particle instance of the 1st particle
# @param Particle q   Particle instance of the 2nd particle
# @return complex     the spinor product [pq]
#
def spinor_prod2(p, q):
    return p.λ_tilda_pa_dot.dot(q.λ_tilda_p__a_dot)[0][0]


# Implementation of the Parker Taylor formula
#
# @param Particle[] gluons an array of n massless gluons with p1 + ... + pn = 0 and pi^2 = 0
# @return complex the amplitude according to the PT formula
#
def PT(gluons):
    # extract negative helicities
    helicities = [o.helicity for o in gluons]
    get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    h_neg = get_indexes(-1, helicities)

    # sanity check for helicities
    if (len(h_neg) != 2):
        warnings.warn("There are not exactly 2 negative helicities in the given gluons; returning NaN.")
        return float('NaN')

    j, k = h_neg[0], h_neg[1]
    n = len(gluons)

    # <jk>^4
    numerator = spinor_prod(gluons[j], gluons[k])**4

    # <12><23> ... <n-1,n>
    denominator = 1.0 + 0j
    for i in range(n - 1):
        denominator *= spinor_prod(gluons[i], gluons[i+1])

    # <n,1>
    denominator *= spinor_prod(gluons[n-1], gluons[0])

    return 1j*numerator/denominator

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# A simple test print function to give colored guidlines
#
# @param string message The test message
# @param bool condition pass/fail True/False
# @return bool pass/fail
#
def test(message, condition):
    print('{4}[TEST]{3} {5}{0}{3} ? {1}{2}{3}'.format(
        message,
        bcolors.OKGREEN if condition[0] else bcolors.FAIL,
        condition if condition[0] else condition,
        bcolors.ENDC,
        bcolors.OKBLUE,
        bcolors.BOLD
    ))
    return condition

# gives the sum pf p's = p_i + ... + p_j
def P_ij(P,i,j):
    return np.sum(P[i:j+1], axis=0)

# (p_i + ... + p_j)^2
def P_ij__2(P,i,j):
    return md(P_ij(P,i,j), P_ij(P,i,j))

# J_μ column vector
def J_a(gluons, P, rec = 1, it = "mu"):
    return J__a(gluons, P, rec, it).dot(g_ab).reshape(4,1)

# J^μ row vector
def J__a(gluons, P, rec = 1, it = "mu"):
    # Off-shell current J^μ(1, ..., n)
    #
    # @param array gluons   array of gluon Particle objects
    # @param array P        array of 4-momenta
    # @param int rec        recursion depth
    # @param int it         name of Lorentz index
    # @return array         row vector
    #
    n = len(gluons)

    # recursion base
    if n == 1:
        return gluons[0].eps__a # row vector
    
    ret = np.zeros(4, dtype=complex)

    # 1st recursion step from i=0 to n-2
    for i in range(0, n-1):
        P_0i = P_ij(P,0,i)
        P_ip1n = P_ij(P,i+1,n)

        J_nu  = J_a(gluons[0:i+1], P[0:i+1], rec+1, "nu")
        J_rho = J_a(gluons[i+1:], P[i+1:], rec+1, "rho")

        for mu in range(0, 4): # mu = 0,1,2,3
            for nu in range(0, 4): # nu = 0,1,2,3                
                for rho in range(0, 4): #rho = 0,1,2,3
                    ret[mu] += V3(mu, nu, rho, P_0i, P_ip1n)*J_nu[nu,0]*J_rho[rho,0]

    # 2nd recursion step from i=0 to n-3 (including n-3) ...
    for i in range(0, n-2):
        # ... and j=i+1 to n-2 (including n-2)
        for j in range(i+1, n-1):
            J_nu    = J_a(gluons[0:i+1],   P[0:i+1],   rec+1, "nu")
            J_rho   = J_a(gluons[i+1:j+1], P[i+1:j+1], rec+1, "rho")
            J_sigma = J_a(gluons[j+1:],    P[j+1:],    rec+1, "sigma")

            for mu in range(0, 4): # mu = 0,1,2,3
                for nu in range(0, 4): # nu = 0,1,2,3                
                    for rho in range(0, 4): #rho = 0,1,2,3
                        for sigma in range(0, 4): #rho = 0,1,2,3
                            ret[mu] += V4(mu, nu, rho, sigma)*J_nu[nu,0]*J_rho[rho,0]*J_sigma[sigma,0]

    # if we are not in the first recursion step
    if rec != 1:
        ret *= -1j/P_ij__2(P, 0, n)

    return ret.reshape(1,4)


def V3(mu, nu, rho, P, Q):
    # Feynman rule for the 3-gluon vertex
    #
    # @param array P      4-momentum of gluon 1
    # @param array Q      4-momentum of gluon 2
    # @param int mu       Lorentz index of gluon 1 (can have values 0,1,2,3)
    # @param int nu       Lorentz index of gluon 2 (can have values 0,1,2,3)
    # @param int rho      Lorentz index of gluon 3 (can have values 0,1,2,3)
    # @return complex
    #
    #     P,μ
    #      |
    #     / \
    # P+Q,λ   Q,ν
    #
    return (1j/np.sqrt(2))*(g__ab[nu][rho]*(P-Q)[mu] + 2*g__ab[rho][mu]*Q[nu] - 2*g__ab[mu][nu]*P[rho])

def V4(mu, nu, rho, sigma):
    # Feynman rule for the 4-gluon vertex
    #
    # @param int mu     Lorentz index of gluon 1
    # @param int nu     Lorentz index of gluon 2
    # @param int rho    Lorentz index of gluon 3
    # @param int sigma  Lorentz index of gluon 4
    # @return complex
    #
    #  μ    ν
    #    \/
    #    /\
    #  λ    ρ
    #
    return 0.5j*( 2*g__ab[mu][rho]*g__ab[nu][sigma] - g__ab[mu][nu]*g__ab[rho][sigma] - g__ab[mu][sigma]*g__ab[nu][rho] )

# J_μ column vector
def J_a_2(gluons, P, rec = 1, it = "mu"):
    return J__a_2(gluons, P, rec, it).dot(g_ab).reshape(4,1)

# J^μ row vector
# J^μ with a clever reference momentum choice: q_1 = p_2, q_2 = ... = q_n = p_1
# V4 and 1st term is V3 vanish 
def J__a_2(gluons, P, rec = 1, it = "mu"):
    n = len(gluons)

    # recursion base
    if n == 1:
        return gluons[0].eps__a # row vector
    
    ret = np.zeros(4, dtype=complex)

    # 1st recursion step from i=0 to n-2
    for i in range(0, n-1):
        P_0i = P_ij(P,0,i)
        P_ip1n = P_ij(P,i+1,n)

        J_nu  = J_a_2(gluons[0:i+1], P[0:i+1], rec+1, "nu")
        J_rho = J_a_2(gluons[i+1:], P[i+1:], rec+1, "rho")

        for mu in range(0, 4): # mu = 0,1,2,3
            for nu in range(0, 4): # nu = 0,1,2,3                
                for rho in range(0, 4): #rho = 0,1,2,3
                    ret[mu] += V3_2(mu, nu, rho, P_0i, P_ip1n)*J_nu[nu,0]*J_rho[rho,0]

    # if we are not in the first recursion step
    if rec != 1:
        ret *= -1j/P_ij__2(P, 0, n)

    return ret.reshape(1,4)

def V3_2(mu, nu, rho, P, Q):
    return 1j*np.sqrt(2)*(g__ab[rho][mu]*Q[nu] - g__ab[mu][nu]*P[rho])

def contract(a, b):
    # Contraction of 2 tensors a and b
    # a^μ * b_μ
    #
    # @param array a tensor a
    # @param array b tensor b
    # @return contraction of a and b
    #

    column = (4,1) # column vec => [[w],[x],[y],[z]]
    row    = (1,4) # row vector => [[w, x, y, z]]
    matrix = (4,4)

    # two vectors -> return a number
    if (a.shape == row and b.shape == column):
        return a.dot(b)[0,0]

    # contraction of tensor with vector -> return a row vector
    if (a.shape == matrix and b.shape == column):
        return a.dot(b).reshape(row)

    # contraction of vector with tensor -> return a row vector
    if (a.shape == row and b.shape == matrix):
        return a.dot(b).reshape(column)

    # contraction of tensor with tensor -> return a tensor
    if (a.shape == matrix and b.shape == matrix):
        return a.dot(b)

    raise ValueError('Dimensions of a and/or b are wrong.')
