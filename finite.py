import numpy as np
from scipy.special import factorial
from scipy import sparse

class UniformPeriodicGrid:

    def __init__(self, N, length):
        self.values = np.linspace(0, length, N, endpoint=False)
        self.dx = self.values[1] - self.values[0]
        self.length = length
        self.N = N


class NonUniformPeriodicGrid:

    def __init__(self, values, length):
        self.values = values
        self.length = length
        self.N = len(values)


class Difference:

    def __matmul__(self, other):
        return self.matrix@other


class DifferenceUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        # set up b
        if (derivative_order + convergence_order)%2 == 0:
            b = np.zeros(derivative_order + convergence_order + 1)
            b[derivative_order] = 1
            kmax = derivative_order + convergence_order
            jmin = -int(kmax/2)
            jmax = int(kmax/2)
        else:
            b = np.zeros(derivative_order + convergence_order)
            b[derivative_order] = 1
            # k \in [0, kmax]
            kmax = derivative_order + convergence_order - 1
            # j \in [jmin, jmax]
            jmin = -int(kmax/2)
            jmax = int(kmax/2)
        # S dimensions JXK
        S = np.zeros((kmax+1,jmax-jmin+1))
        for k in range(0,kmax+1):
            for j in range(jmin,jmax+1):
                S[k,j-jmin] = 1/factorial(k)*(j*grid.dx)**k
        # solve for a
        test = np.array([[ 1, 1, 1],
              [-grid.dx, 0, grid.dx],
              [grid.dx**2/2, 0, grid.dx**2/2]])
        a = np.linalg.inv(S) @ b
        shape = [grid.N, grid.N]
        stencil = a
        offsets = range(jmin,jmax+1)
        D = sparse.diags(stencil, offsets=offsets, shape=shape)
        D = D.tocsr()
        # upper right
        for i in range(jmax):
            for j in range(jmin+i,0):
                D[i,j] = stencil[j-(jmin+i)]
        # lower left
        for i in range(grid.N-jmax,grid.N):
            for j in range(0,i-(grid.N-jmax)+1):
                D[i,j] = stencil[jmax+grid.N-i+j]
        self.matrix = D
        pass


class DifferenceNonUniformGrid(Difference):

    def __init__(self, derivative_order, convergence_order, grid, axis=0, stencil_type='centered'):

        self.derivative_order = derivative_order
        self.convergence_order = convergence_order
        self.stencil_type = stencil_type
        self.axis = axis
        pass
