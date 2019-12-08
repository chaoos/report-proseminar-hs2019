import numpy as np

g = np.diag([1.0, -1.0, -1.0, -1.0])

#
# class BGCurrent
#
class BGCurrent:
    #
    # @param BGCurrent self     reference to the class instance
    # @param array gluons       array of gluon Particle objects
    # @return BGCurrent         class instance
    #
    def __init__ (self, gluons):
        self.gluons = gluons
        self.n = gluons.size
        self.regJ = np.zeros((self.n, self.n, 4), dtype=complex)
        self.memoryJ = np.zeros((self.n, self.n), dtype=bool) # filled with False
        self.vertex4 = v4()
        self.P = np.array([g.p for g in self.gluons])
        #self.registerP_ij = np.zeros((self.n, self.n, 4), dtype=complex)
        #self.memoryP_ij = np.zeros((self.n, self.n), dtype=bool) # filled with False


    def J_a(self, a = 0, b = None, rec = 1):
        return np.einsum('ij,i->i', g, self.J__a(a, b, rec))

    def J__a(self, a = 0, b = None, rec = 1):
        #
        # @param BGCurrent self    reference to the class instance
        # @param int a             starting index
        # @param int|None b        ending index
        # @param int rec           recursion depth
        # @return array            the current as a 4-vector
        #
        
        if b == None:
            b = self.n - 1

        # if this current was already calculated, don't do it again
        if self.memoryJ[a,b] == True:
            return self.regJ[a,b]

        # recursion base
        if a == b:
            self.memoryJ[a,b] = True
            self.regJ[a,b] = self.gluons[a].eps__a.reshape(4)
            return self.regJ[a,b]

        # 1st recursion step from i=0 to n-2
        for i in range(a, b):
            J_nu  = self.J_a(a, i, rec+1)
            J_rho = self.J_a(i+1, b,  rec+1)
            v = v3(self.P_ij(a, i), self.P_ij(i+1, b))
            self.regJ[a,b] += np.einsum('mnr,n,r->m', v, J_nu, J_rho)

        # 2nd recursion step from i=0 to n-3 (including n-3) ...
        for i in range(a, b-1):
            # ... and j=i+1 to n-2 (including n-2)
            for j in range(i+1, b):
                J_nu    = self.J_a(a,   i, rec+1)
                J_rho   = self.J_a(i+1, j, rec+1)
                J_sigma = self.J_a(j+1, b, rec+1)
                self.regJ[a,b] += np.einsum('mnrs,n,r,s->m', self.vertex4, J_nu, J_rho, J_sigma)

        # if we are not in the first recursion step
        # the gluon propagator needs to be appended
        if rec != 1:
            self.regJ[a,b] *= gpg(self.P_ij(a, b))

        self.memoryJ[a,b] = True;
        return self.regJ[a,b]

    def P_ij(self, i, j):
        # @return complex gives the sum p_i + ... + p_j
        return np.sum(self.P[i:j+1], axis=0)
        """
        if self.memoryP_ij[i,j] == True:
            #print("from reg", i,j)
            return self.registerP_ij[i,j]

        self.memoryP_ij[i,j] = True
        self.registerP_ij[i,j] = np.sum(self.P[i:j+1], axis=0)
        return self.registerP_ij[i,j]
        """

def gpg(p):
    # Feynman rule for the gluon propagator
    #
    # @param array p    4-momentum of the virtual particle
    # @return complex   the gluon propagator
    #
    return -1j/np.einsum('i,ij,j', p, g, p)

def v3(P, Q):
    # Feynman rule for the 3-gluon vertex
    #
    # @param array P      4-momentum of gluon 1
    # @param array Q      4-momentum of gluon 2
    # @return ndarray
    #
    #     P,μ
    #      |
    #     / \
    # P+Q,λ   Q,ν
    #
    return (1j/np.sqrt(2))*(
            np.einsum('jk,i', g, P-Q)
        - 2*np.einsum('ij,k', g, P)
        + 2*np.einsum('ki,j', g, Q)
    )

def v4():
    # Feynman rule for the 4-gluon vertex
    #
    # @return ndarray
    #
    #  μ    ν
    #    \/
    #    /\
    #  λ    ρ
    # 
    # i,j,k,l = mu,nu,rho,sigma
    return 0.5j*(
        2*np.einsum('ik,jl', g, g)
        - np.einsum('ij,kl', g, g)
        - np.einsum('il,jk', g, g)
    )