import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
from scipy.special import factorial
from collections import deque
from farray import axslice, apply_matrix

class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.X.gather()
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1

    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.F = eq_set.F
        if hasattr(eq_set, 'BC'):
            self.BC = eq_set.BC
        else:
            self.BC = None

    def step(self, dt):
        super().step(dt)
        if self.BC:
            self.BC(self.X)
            self.X.scatter()


class ImplicitTimestepper(Timestepper):

    def __init__(self, eq_set, axis):
        super().__init__()
        self.axis = axis
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        N = len(self.X.data)
        self.I = sparse.eye(N, N)

    def _LUsolve(self, data):
        if self.axis == 0:
            return self.LU.solve(data)
        elif self.axis == len(data.shape)-1:
            return self.LU.solve(data.T).T
        else:
            raise ValueError("Can only do implicit timestepping on first or last axis")


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.X.data + dt*self.F(self.X)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, eq_set):
        super().__init__(eq_set)
        N = len(X.data)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.X.data + dt*self.F(self.X)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.X_old = np.copy(self.X.data)
            return self.X + dt*self.F(self.X)
        else:
            X_temp = self.X_old + 2*dt*self.F(self.X)
            self.X_old = np.copy(self.X)
            return X_temp


class Multistage(ExplicitTimestepper):

    def __init__(self, eq_set, stages, a, b):
        super().__init__(eq_set)
        self.stages = stages
        self.a = a
        self.b = b

        self.X_list = []
        self.K_list = []
        for i in range(self.stages):
            self.X_list.append(StateVector([np.copy(var) for var in self.X.variables]))
            self.K_list.append(np.copy(self.X.data))

    def _step(self, dt):
        X = self.X
        X_list = self.X_list
        K_list = self.K_list
        stages = self.stages

        np.copyto(X_list[0].data, X.data)
        for i in range(1, stages):
            K_list[i-1] = self.F(X_list[i-1])

            np.copyto(X_list[i].data, X.data)
            # this loop is slow -- should make K_list a 2D array
            for j in range(i):
                X_list[i].data += self.a[i, j]*dt*K_list[j]
            if self.BC:
                self.BC(X_list[i])

        K_list[-1] = self.F(X_list[-1])

        # this loop is slow -- should make K_list a 2D array
        for i in range(stages):
            X.data += self.b[i]*dt*K_list[i]

        return X.data


def RK22(eq_set):
    a = np.array([[  0,   0],
                  [1/2,   0]])
    b = np.array([0, 1])
    return Multistage(eq_set, 2, a, b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        self.f_list = deque()
        for i in range(self.steps):
            self.f_list.append(np.copy(X.data))

    def _step(self, dt):
        f_list = self.f_list
        f_list.rotate()
        f_list[0] = self.F(self.X)
        if self.iter < self.steps:
            coeffs = self._coeffs(self.iter+1)
        else:
            coeffs = self._coeffs(self.steps)

        for i, coeff in enumerate(coeffs):
            self.X.data += self.dt*coeff*self.f_list[i].data
        return self.X.data

    def _coeffs(self, num):
        i = (1 + np.arange(num))[None, :]
        j = (1 + np.arange(num))[:, None]
        S = (-i)**(j-1)/factorial(j-1)

        b = (-1)**(j+1)/factorial(j)

        a = np.linalg.solve(S, b)
        return a


class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(self.X.data)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.M + dt/2*self.L
            self.RHS = self.M - dt/2*self.L
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self._LUsolve(apply_matrix(self.RHS, self.X.data, self.axis))


class BackwardDifferentiationFormula(ImplicitTimestepper):

    def __init__(self, u, L, steps):
        super().__init__(u,L)
        N = len(u)
        self.I = sparse.eye(N, N)
        self.steps = steps

    def _step(self, dt):
        if (self.iter == 0):
        # if (True):
            # print("iter =",self.iter)
            N = len(self.u)
            # initialize u_vec, (s+1)XN
            global u_vec
            u_vec = np.zeros((self.steps+1,N))
            u_vec[0] = np.copy(self.u)
            # initialize A, 1X(s+1)
            global A
            A = np.zeros(self.steps+1)
            # initialize dt_vec, 1X(s+1)
            global dt_vec
            dt_vec = np.zeros(self.steps+1)
            for i in range(self.steps+1):
                dt_vec[i] = -i*dt
            # BackwardEuler
            LHS = self.I - dt*self.L.matrix
            return spla.spsolve(LHS, self.u)
        # first s timesteps
        if (self.iter < self.steps and self.iter > 0):
            S_len = self.iter + 2
            for i in reversed(range(1,self.iter+1)):
                u_vec[i] = np.copy(u_vec[i-1])
            u_vec[0] = np.copy(self.u)
            S = np.zeros((S_len,S_len))
            for i in range(S_len):
                for j in range(S_len):
                    S[i,j] = 1/factorial(j)*(-i*dt)**j
            b = [0]*S_len
            b[1] = 1
            a = b @ np.linalg.inv(S)
            A[0:S_len] = a
            LHS = self.L.matrix - A[0]*self.I
            return spla.spsolve(LHS,A[1:self.iter+2]@u_vec[0:self.iter+1])
        # self.iter >= self.steps
        b = [0]*(self.steps+1)
        b[1] = 1
        # S is factorial coefficient matrix
        # S = np.zeros((self.steps+1,self.steps+1))
        # for i in range(self.steps+1):
        #     for j in range(self.steps+1):
        #         S[i,j] = 1/factorial(j)*(-i*dt)**j
        # rotate dt_vec
        dt_vec = dt_vec - dt
        for i in reversed(range(1,self.steps+1)):
            dt_vec[i] = dt_vec[i-1]
        dt_vec[0] = 0
        # S based on dt_vec
        S = np.zeros((self.steps+1,self.steps+1))
        for i in range(self.steps+1):
            for j in range(self.steps+1):
                S[i,j] = 1/factorial(j)*(dt_vec[i])**j
        # a is the stencil
        a = b @ np.linalg.inv(S)
        A = a
        # replace old row
        for i in reversed(range(1,self.steps+1)):
            u_vec[i] = np.copy(u_vec[i-1])
        u_vec[0] = np.copy(self.u)

        LHS = self.L.matrix - A[0]*self.I
        return spla.spsolve(LHS,A[1:]@u_vec[:-1])
        # if (self.iter == self.steps):
        #     self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        # return self.LU.solve(A[1:]@u_vec[:-1])


class StateVector:

    def __init__(self, variables, axis=0):
        self.axis = axis
        var0 = variables[0]
        shape = list(var0.shape)
        self.N = shape[axis]
        shape[axis] *= len(variables)
        self.shape = tuple(shape)
        self.data = np.zeros(shape)
        self.variables = variables
        self.gather()

    def gather(self):
        for i, var in enumerate(self.variables):
            np.copyto(self.data[axslice(self.axis, i*self.N, (i+1)*self.N)], var)

    def scatter(self):
        for i, var in enumerate(self.variables):
            np.copyto(var, self.data[axslice(self.axis, i*self.N, (i+1)*self.N)])


class IMEXTimestepper(Timestepper):

    def __init__(self, eq_set):
        super().__init__()
        self.X = eq_set.X
        self.M = eq_set.M
        self.L = eq_set.L
        self.F = eq_set.F

    def step(self, dt):
        self.X.data = self._step(dt)
        self.X.scatter()
        self.dt = dt
        self.t += dt
        self.iter += 1


class Euler(IMEXTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            LHS = self.M + dt*self.L
            self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')
        
        RHS = self.M @ self.X.data + dt*self.F(self.X)
        return self.LU.solve(RHS)


class CNAB(IMEXTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            # Euler
            LHS = self.M + dt*self.L
            LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data + dt*self.FX
            self.FX_old = self.FX
            return LU.solve(RHS)
        else:
            if dt != self.dt or self.iter == 1:
                LHS = self.M + dt/2*self.L
                self.LU = spla.splu(LHS.tocsc(), permc_spec='NATURAL')

            self.FX = self.F(self.X)
            RHS = self.M @ self.X.data - 0.5*dt*self.L @ self.X.data + 3/2*dt*self.FX - 1/2*dt*self.FX_old
            self.FX_old = self.FX
            return self.LU.solve(RHS)


class BDFExtrapolate(IMEXTimestepper):

    def __init__(self, eq_set, steps):
        super().__init__(eq_set)
        self.steps = steps
        pass

    def _step(self, dt):
        # Euler method on step 0
        if (self.iter == 0):
            # store X from X[n-s] to X[n]
            global X_store
            X_store = [0*self.X.data for i in range(self.steps+1)]
            # FX_store stores FX[n-s] to FX[n-1]
            global FX_store
            FX_store = [0*self.F(self.X) for i in range(self.steps)]
            LHS = self.M + dt*self.L
            FX_new = self.F(self.X)
            RHS = self.M@self.X.data + dt*FX_new
            FX_store[0] = np.copy(FX_new)
            X_store[0] = np.copy(self.X.data)
            return spla.spsolve(LHS,RHS)
        if (self.iter < self.steps and self.iter > 0):
            # update FX_store
            for i in reversed(range(1,self.iter+1)):
                FX_store[i] = np.copy(FX_store[i-1])
            FX_new = self.F(self.X)
            FX_store[0] = np.copy(FX_new)
            # update X_store
            for i in reversed(range(1,self.iter+1)):
                X_store[i] = np.copy(X_store[i-1])
            X_store[0] = np.copy(self.X.data)
            LHS = self.M + dt*self.L
            RHS = self.M@self.X.data + dt*FX_new
            return spla.spsolve(LHS,RHS)
        # Euler if you like
        # FX_new = self.F(self.X)
        # LHS = self.M + dt*self.L
        # RHS = self.M@self.X.data + dt*FX_new
        # return spla.spsolve(LHS,RHS)
        # find coefficients of a
        b_RHS = [0]*(self.steps+1)
        b_RHS[1] = 1
        S = np.zeros((self.steps+1,self.steps+1))
        for i in range(self.steps+1):
            for j in range(self.steps+1):
                S[i,j] = 1/factorial(j)*(-i*dt)**j
        # coefficients of a, from a[0] to a[s]
        a = b_RHS @ np.linalg.inv(S)
        # find coefficients of b, b[1] to b[s]
        b_RHS2 = [0]*self.steps 
        b_RHS2[0] = 1
        S2 = np.zeros((self.steps,self.steps))
        for i in range(self.steps):
            for j in range(self.steps):
                S2[i,j] = 1/factorial(j)*(-(i+1)*dt)**(j)
        b = b_RHS2 @ np.linalg.inv(S2)
        # update FX_store
        for i in reversed(range(1,self.steps)):
            FX_store[i] = np.copy(FX_store[i-1])
        FX_store[0] = self.F(self.X)
        # update X_store
        for i in reversed(range(1,self.steps+1)):
            X_store[i] = np.copy(X_store[i-1])
        X_store[0] = np.copy(self.X.data)
        LHS = a[0] * self.M + self.L 
        RHS = - self.M @ (a[1:]@X_store[:-1]) + b@FX_store
        return spla.spsolve(LHS,RHS)
