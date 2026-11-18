"""Scalar two-step model extracted from the supplied notebook. See PROVENANCE.md."""
import numpy as np
from numpy.polynomial import Polynomial

class system:
    def __init__(self, params_y, params_x, params_LQR,
                 mu0=0.1, Sigma0=2.0):
        self.C0, self.C1, self.Znoise = params_y
        self.A, self.B, self.Pnoise = params_x
        self.Q, self.R, self.QT = params_LQR
        self.mu0, self.Sigma0 = mu0, Sigma0

onedcase = system(
    params_y=(0.0, 1.0, 0.09),     # C1 后面设为 alpha；C0 用于扫描
    params_x=(0.9, 1.0, 0.01),
    params_LQR=(1.0, 1.0, 1.0)
)



def Compute_Kt_Pt(s):
    K, P = np.zeros(3), np.zeros(2)
    K[2] = s.QT
    for t in (1, 0):
        P[t] = (s.A*s.B*K[t+1])**2 / (s.B**2*K[t+1] + s.R)
        K[t] = s.Q + s.A**2*K[t+1] - P[t]
    return K, P

s = onedcase
Kt, Pt = Compute_Kt_Pt(s)

alpha = s.B**2*Kt[1] + s.R
beta = s.B*Kt[1]*s.A
gamma = s.Znoise*s.A**2*Pt[1]
kappa = s.Znoise/s.Sigma0

s.C1 = alpha
u_sep = -beta*s.mu0/alpha

def objective(u, C0, C1, mu=s.mu0):
    u = np.asarray(u)
    f_LQG = alpha*u**2 + 2*beta*mu*u
    g = gamma / ((C0 + C1*u)**2 + kappa)
    return f_LQG, g, f_LQG + g

def argmin_objective(C0, C1, mu=s.mu0):
    u = Polynomial([0., 1.])
    c = C0 + C1*u

    equation = (alpha*u + beta*mu)*(c*c + kappa)**2 - gamma*C1*c
    roots = equation.roots()
    roots = roots[np.abs(roots.imag) < 1e-7].real

    values = objective(roots, C0, C1, mu)[2]
    best = np.isclose(values, values.min(), rtol=1e-9, atol=1e-10)
    return np.sort(roots[best])
