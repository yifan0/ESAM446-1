from timesteppers import StateVector, CrankNicolson, RK22
import finite
from scipy import sparse
import numpy as np

class ViscousBurgers:
    
    def __init__(self, u, nu, d, d2):
        self.u = u
        self.X = StateVector([u])
        
        N = len(u)
        self.M = sparse.eye(N, N)
        self.L = -nu*d2.matrix
        
        f = lambda X: -X.data*(d @ X.data)
        
        self.F = f


class Wave:
    
    def __init__(self, u, v, d2):
        self.X = StateVector([u, v])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))

        M00 = I
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                              [M10, M11]])

        L00 = Z
        L01 = -I
        L10 = -d2.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                              [L10, L11]])

        self.F = lambda X: 0*X.data


class SoundWave:

    def __init__(self, u, p, d, rho0, gammap0):
        self.X = StateVector([u, p])
        N = len(u)
        I = sparse.eye(N, N)
        Z = sparse.csr_matrix((N, N))
        if np.isscalar(rho0):
            M00 = rho0*I
        else:
            M00 = np.diag(rho0)
        M01 = Z
        M10 = Z
        M11 = I
        self.M = sparse.bmat([[M00, M01],
                            [M10, M11]])

        L00 = Z
        L01 = d.matrix
        if np.isscalar(gammap0):
            L10 = gammap0*d.matrix
        else:
            L10 = np.diag(gammap0)@d.matrix
        L11 = Z
        self.L = sparse.bmat([[L00, L01],
                            [L10, L11]])

        self.F = lambda X: 0*X.data


class ReactionDiffusion:
    
    def __init__(self, c, d2, c_target, D):
        self.X = StateVector([c])
        N = len(c)

        self.M = sparse.eye(N, N)
        self.L = -D*d2.matrix

        self.F = lambda c: c.data*(c_target-c.data)

class ReactionDiffusion2D:

    def __init__(self, c, D, dx2, dy2):
        self.t = 0
        self.iter = 0
        self.dt = None
        class Diffusionx:
            def __init__(self, c, D, d2x):
                self.X = StateVector([c], axis=0)
                N = c.shape[0]
                self.M = sparse.eye(N, N)
                self.L = -D*d2x.matrix


        class Diffusiony:
            def __init__(self, c, D, d2y):
                self.X = StateVector([c], axis=1)
                N = c.shape[1]
                self.M = sparse.eye(N, N)
                self.L = -D*d2y.matrix
        diffx = Diffusionx(c,D,dx2)
        diffy = Diffusiony(c,D,dy2)
        class Reaction:
            def __init__(self, c):
                self.X = StateVector([c])
                N = c.shape[1]
                self.M = sparse.eye(N, N)
                self.L = lambda X: 0*X.data
                self.F = lambda X: X.data*(1-X.data)
        self.ts_c = RK22(Reaction(c))
        self.ts_x = CrankNicolson(diffx,0)
        self.ts_y = CrankNicolson(diffy,1)
        pass
    

    def step(self, dt):
        # self.ts_x.step(dt)
        # self.ts_y.step(dt)
        # self.ts_c.step(dt)
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_c.step(dt/2)
        self.ts_c.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        
        self.t += dt
        self.iter += 1
        pass


class ViscousBurgers2D:

    def __init__(self, u, v, nu, spatial_order, domain):
        pass

    def step(self, dt):
        pass