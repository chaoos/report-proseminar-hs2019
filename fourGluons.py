from helpers import *
import itertools

def fourGluonScattering(gluons):
    M1, M1_2, M1_3 = sTypeContribution(gluons)
    M2 = pTypeContribution(gluons)
    M3, M3_2 = fourVertexContribution(gluons)
    return M1 + M2 + M3, M1, M2, M3, M1_2, M1_3, M3_2


'''
2    3
 \__/
 /  \
1    4
s-type
'''
def sTypeContribution(gluons):

    # give normal array instead of row vectors (2d arrays)
    e1 = gluons[0].ε__a.reshape(4)
    e2 = gluons[1].ε__a.reshape(4)
    e3 = gluons[2].ε__a.reshape(4)
    e4 = gluons[3].ε__a.reshape(4)
    e_p1_a = gluons[0].ε_a.reshape(4)
    e_p2_a = gluons[1].ε_a.reshape(4)
    e_p3_a = gluons[2].ε_a.reshape(4)
    e_p4_a = gluons[3].ε_a.reshape(4)
    e_p1__a = gluons[0].ε__a.reshape(4)
    e_p2__a = gluons[1].ε__a.reshape(4)
    e_p3__a = gluons[2].ε__a.reshape(4)
    e_p4__a = gluons[3].ε__a.reshape(4)

    p1, p2, p3, p4 = gluons[0].p, gluons[1].p, gluons[2].p, gluons[3].p

    pi = p1 + p2
    pi_squared = minkowskiDot(pi, pi)
    #for λ, μ, σ, τ, ν, ρ in itertools.product(x, x, x, x, x, x):
    #    M += e_p1[λ]*V3g(p2, -p, p1, μ, σ, λ)*e_p2[μ]*Pg(p, σ, τ)*e_p3[ν]*V3g(p4, p3, -p, ρ, τ, ν)*e_p4[ρ]
    M1 = 0.0
    M1 += e_p3__a.dot(e_p4_a)*(p3-p4).dot(e_p1_a)*(p1+pi).dot(e_p2_a)     # (1)
    M1 += e_p1__a.dot(e_p4_a)*(p1+pi).dot(e_p2_a)*(p4+pi).dot(e_p3_a)      # (2)
    M1 += e_p1__a.dot(e_p3_a)*(p1+pi).dot(e_p2_a)*(-p3-pi).dot(e_p4_a)     # (3)
    M1 += e_p3__a.dot(e_p4_a)*(-p2-pi).dot(e_p1_a)*(p3-p4).dot(e_p2_a)    # (4)
    M1 += e_p2__a.dot(e_p4_a)*(-p2-pi).dot(e_p1_a)*(p4+pi).dot(e_p3_a)     # (5)
    M1 += e_p2__a.dot(e_p3_a)*(-p2-pi).dot(e_p1_a)*(-p3-pi).dot(e_p4_a)    # (6)
    M1 += e_p1__a.dot(e_p2_a)*e_p3__a.dot(e_p4_a)*minkowskiDot((p1-p2), (p3-p4))  # (7)
    M1 += e_p1__a.dot(e_p2_a)*(p4+pi).dot(e_p3_a)*(p1-p2).dot(e_p4_a)     # (8)
    M1 += e_p1__a.dot(e_p2_a)*(p1-p2).dot(e_p3_a)*(-p3-pi).dot(e_p4_a)    # (9)
    M1 *= (0.5j/pi_squared)

    # 2nd way to calculate this
    M1_2 = 0.0
    M1_2 += md(e3, e4)*md((p3-p4 ), e1)*md((p1+pi ), e2)    # (1)
    M1_2 += md(e1, e4)*md((p1+pi ), e2)*md((p4+pi ), e3)    # (2)
    M1_2 += md(e1, e3)*md((p1+pi ), e2)*md((-p3-pi), e4)    # (3)
    M1_2 += md(e3, e4)*md((-p2-pi), e1)*md((p3-p4 ), e2)    # (4)
    M1_2 += md(e2, e4)*md((-p2-pi), e1)*md((p4+pi ), e3)    # (5)
    M1_2 += md(e2, e3)*md((-p2-pi), e1)*md((-p3-pi), e4)    # (6)
    M1_2 += md(e1, e2)*md(e3, e4)*md((p1-p2), (p3-p4))      # (7)
    M1_2 += md(e1, e2)*md((p4+pi ), e3)*md((p1-p2 ), e4)    # (8)
    M1_2 += md(e1, e2)*md((p1-p2 ), e3)*md((-p3-pi), e4)    # (9)
    M1_2 *= (0.5j/pi_squared)

    # from https://www.ttp.kit.edu/~melnikov/TTP2/notes/notes18.pdf
    M1_3 = 0.0
    M1_3 += md(e1, e2)*md(e3, e4)*md((p1-p2), (p3-p4))    # (7)
    M1_3 += md(e1, e2)*md((p4+pi ), e3)*md((p1-p2 ), e4)  # (8)
    M1_3 += md(e1, e2)*md((p1-p2 ), e3)*md((-p3-pi), e4)  # (9)
    M1_3 += md(e3, e4)*md((p2+pi ), e1)*md((p3-p4 ), e2)  # (4) differs + <-> -
    M1_3 += md(e2, e4)*md((p2+pi ), e1)*md((p4+pi ), e3)  # (5) differs + <-> -
    M1_3 += md(e2, e3)*md((p2+pi ), e1)*md((-p3-pi), e4)  # (6) differs + <-> -
    M1_3 += md(e3, e4)*md((p3-p4 ), e1)*md((-p1-pi), e2)  # (1) differs + <-> -
    M1_3 += md(e1, e4)*md((-p1-pi), e2)*md((p4+pi ), e3)  # (2) differs + <-> -
    M1_3 += md(e1, e3)*md((-p1-pi), e2)*md((-p3-pi), e4)  # (3) differs + <-> -
    M1_3 *= (0.5j/pi_squared)

    return M1, M1_2, M1_3


'''
2   3
 \ /
  |
 / \
1   4
t-type
'''
def pTypeContribution(gluons):

    # give normal array instead of row vectors (2d arrays)
    e1 = gluons[0].ε__a.reshape(4)
    e2 = gluons[1].ε__a.reshape(4)
    e3 = gluons[2].ε__a.reshape(4)
    e4 = gluons[3].ε__a.reshape(4)

    p1, p2, p3, p4 = gluons[0].p, gluons[1].p, gluons[2].p, gluons[3].p

    pi = p2 + p3
    pi_squared = minkowskiDot(pi, pi)
    #for μ, ν, λ, ρ, τ, σ in itertools.product(x, x, x, x, x, x):
    #    M += e_p1[λ]*V3g(p1, -p, p4, λ, σ, ρ)*e_p4[ρ]*Pg(p, σ, τ)*e_p2[μ]*V3g(p3, -p, p2, ν, τ, μ)*e_p3[ν]
    # from https://www.ttp.kit.edu/~melnikov/TTP2/notes/notes18.pdf
    M2 = 0.0
    M2 += md(e2, e3)*md(e1, e4)*md((p2-p3), (p4-p1))     # (1)
    M2 += md(e2, e3)*md((p1+pi ), e4)*md((p2-p3 ), e1)   # (2)
    M2 += md(e2, e3)*md((-p4-pi), e1)*md((p2-p3 ), e4)   # (3)
    M2 += md(e1, e4)*md((p3+pi ), e2)*md((p4-p1 ), e3)   # (4)
    M2 += md(e1, e3)*md((p3+pi ), e2)*md((p1+pi ), e4)   # (5)
    M2 += md(e3, e4)*md((p3+pi ), e2)*md((-p4-pi), e1)   # (6)
    M2 += md(e1, e4)*md((-p2-pi), e3)*md((p4-p1 ), e2)   # (7)
    M2 += md(e1, e2)*md((-p2-pi), e3)*md((p1+pi ), e4)   # (8)
    M2 += md(e2, e4)*md((-p2-pi), e3)*md((-p4-pi), e1)   # (9)
    M2 *= (0.5j/pi_squared)

    return M2

'''
2  3
 \/
 /\
1  4
4-gluon vertex
'''
def fourVertexContribution(gluons):

    e_p1_a = gluons[0].ε_a.reshape(4)
    e_p2_a = gluons[1].ε_a.reshape(4)
    e_p3_a = gluons[2].ε_a.reshape(4)
    e_p4_a = gluons[3].ε_a.reshape(4)
    e_p1__a = gluons[0].ε__a.reshape(4)
    e_p2__a = gluons[1].ε__a.reshape(4)
    e_p3__a = gluons[2].ε__a.reshape(4)
    e_p4__a = gluons[3].ε__a.reshape(4)

    M3 = 0.0
    M3 +=   1j*e_p1__a.dot(e_p3_a)*e_p2__a.dot(e_p4_a)
    M3 -= 0.5j*e_p1__a.dot(e_p2_a)*e_p3__a.dot(e_p4_a)
    M3 -= 0.5j*e_p1__a.dot(e_p4_a)*e_p2__a.dot(e_p3_a)

    x = [0, 1, 2, 3] # the Minkowski indices

    M3_2 = 0.0
    for λ, μ, ν, ρ in itertools.product(x, x, x, x):
        #M3_alt += e_p1[λ]*e_p2[μ]*V4g(λ, μ, ν, ρ)*e_p3[ν]*e_p4[ρ]
        M3_2 +=    1j*e_p1__a[λ]*g[λ][ν]*e_p3_a[ν]*e_p2__a[μ]*g[μ][ρ]*e_p4_a[ρ] # 1. Term
        M3_2 += -0.5j*e_p1__a[λ]*g[λ][μ]*e_p2_a[μ]*e_p3__a[ν]*g[ν][ρ]*e_p4_a[ρ] # 2. Term
        M3_2 += -0.5j*e_p1__a[λ]*g[λ][ρ]*e_p4_a[ρ]*e_p2__a[μ]*g[μ][ν]*e_p3_a[ν] # 3. Term

    return M3, M3_2