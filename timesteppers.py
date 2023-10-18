import numpy as np
from scipy.special import factorial
from scipy import sparse
import scipy.sparse.linalg as spla
class Timestepper:

    def __init__(self):
        self.t = 0
        self.iter = 0
        self.dt = None

    def step(self, dt):
        self.u = self._step(dt)
        self.t += dt
        self.iter += 1
        
    def evolve(self, dt, time):
        while self.t < time - 1e-8:
            self.step(dt)


class ExplicitTimestepper(Timestepper):

    def __init__(self, u, f):
        super().__init__()
        self.u = u
        self.f = f


class ForwardEuler(ExplicitTimestepper):

    def _step(self, dt):
        return self.u + dt*self.f(self.u)


class LaxFriedrichs(ExplicitTimestepper):

    def __init__(self, u, f):
        super().__init__(u, f)
        N = len(u)
        A = sparse.diags([1/2, 1/2], offsets=[-1, 1], shape=[N, N])
        A = A.tocsr()
        A[0, -1] = 1/2
        A[-1, 0] = 1/2
        self.A = A

    def _step(self, dt):
        return self.A @ self.u + dt*self.f(self.u)


class Leapfrog(ExplicitTimestepper):

    def _step(self, dt):
        if self.iter == 0:
            self.u_old = np.copy(self.u)
            return self.u + dt*self.f(self.u)
        else:
            u_temp = self.u_old + 2*dt*self.f(self.u)
            self.u_old = np.copy(self.u)
            return u_temp


class LaxWendroff(Timestepper):

    def __init__(self, u, f1, f2):
        super().__init__()
        self.u = u
        self.f1 = f1
        self.f2 = f2

    def _step(self, dt):
        return self.u + dt*self.f1(self.u) + dt**2/2*self.f2(self.u)


class Multistage(ExplicitTimestepper):

    def __init__(self, u, f, stages, a, b):
        super().__init__(u, f)
        self.stages = stages
        self.a = a
        self.b = b

    def _step(self, dt):
        # k_list is N X stages
        k_list = [[0]*np.size(self.u) for i in range(self.stages)]
        # fill k_list stage by stage
        for i in range(0,self.stages):
            # dot product of a[i] and k_list gives next stage k
            k_list[i] = self.f(self.u + dt*np.dot(self.a[i],k_list))
        # matrix product of k^T and b
        return self.u + dt*np.matmul(np.transpose(k_list),self.b)


class AdamsBashforth(ExplicitTimestepper):

    def __init__(self, u, f, steps, dt):
        super().__init__(u, f)
        self.steps = steps
        self.dt = dt

    def _step(self, dt):
        # print("iteration =", self.iter)
        N = np.size(self.u)
        # print("IC =",self.u)
        if (self.iter == 0):
            # initialize list of f(u^n)
            global f_vec
            f_vec = np.zeros((self.steps,N))
            f_vec[0] = np.copy(self.f(self.u))
            # initialize A
            global A
            A = np.zeros(self.steps)
            S_len = self.iter + 1
            S = np.zeros((S_len,S_len))
            for i in range(S_len):
                for j in range(S_len):
                    S[i,j] = 1/factorial(j)*(-i*dt)**j
            b = [0]*S_len
            b[0] = dt
            a = b @ np.linalg.inv(S)
            A[0] = a
            return self.u + A @ f_vec

        # first (s-1) timesteps
        if (self.iter < self.steps and self.iter > 0):
            S_len = self.iter + 1
            for i in reversed(range(1,self.iter+1)):
                f_vec[i] = np.copy(f_vec[i-1])
            f_vec[0] = np.copy(self.f(self.u))
            S = np.zeros((S_len,S_len))
            for i in range(S_len):
                for j in range(S_len):
                    S[i,j] = 1/factorial(j)*(-i*dt)**j
            b = [0]*S_len
            for i in range(S_len):
                b[i] = 1/factorial(i+1)*dt**(i+1)
            a = b @ np.linalg.inv(S)
            A[0:S_len] = a
            return self.u + A @ f_vec

        # b = [dt,dt**2/2!,...dt**s/s!]
        b = [0]*self.steps
        for i in range(self.steps):
            b[i] = 1/factorial(i+1)*dt**(i+1)
        
        # S is factorial coefficient matrix
        S = np.zeros((self.steps,self.steps))
        for i in range(self.steps):
            for j in range(self.steps):
                S[i,j] = 1/factorial(j)*(-i*dt)**j
        # a is the stencil
        a = b @ np.linalg.inv(S)
        A = a
        # replace old row
        for i in reversed(range(1,self.steps)):
            f_vec[i] = np.copy(f_vec[i-1])
        f_vec[0] = np.copy(self.f(self.u))
        return self.u + A @ f_vec


class ImplicitTimestepper(Timestepper):

    def __init__(self, u, L):
        super().__init__()
        self.u = u
        self.L = L
        N = len(u)
        self.I = sparse.eye(N, N)

class BackwardEuler(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.u)


class CrankNicolson(ImplicitTimestepper):

    def _step(self, dt):
        if dt != self.dt:
            self.LHS = self.I - dt/2*self.L.matrix
            self.RHS = self.I + dt/2*self.L.matrix
            self.LU = spla.splu(self.LHS.tocsc(), permc_spec='NATURAL')
        return self.LU.solve(self.RHS @ self.u)


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
