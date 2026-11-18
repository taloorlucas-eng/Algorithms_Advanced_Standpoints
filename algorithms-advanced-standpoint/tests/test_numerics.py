"""Independent numerical checks; these do not validate full training pipelines."""
from pathlib import Path
import sys
import importlib.util
import unittest
import numpy as np
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'studies/separation_gap'))
import model as gap
from run import simulate_costs


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SeparationChecks(unittest.TestCase):
    def test_symmetric_solution_matches_polynomial_roots(self):
        c1=gap.s.C1; c0=-c1*gap.u_sep
        radius=np.sqrt(abs(c1)*np.sqrt(gap.gamma/gap.alpha)-gap.kappa)/abs(c1)
        np.testing.assert_allclose(gap.argmin_objective(c0,c1),gap.u_sep+np.array([-radius,radius]),atol=1e-10)

    def test_root_objectives_match_dense_grid_for_all_offsets(self):
        u=np.linspace(-4,4,200001)
        for delta in (1.,0.,-.5,-1.,-2.):
            c1=gap.s.C1;c0=-c1*(gap.u_sep+delta)
            roots=gap.argmin_objective(c0,c1)
            best=float(gap.objective(roots[0],c0,c1)[2])
            grid=float(gap.objective(u,c0,c1)[2].min())
            self.assertLessEqual(best,grid+1e-10)
            self.assertLess(grid-best,1e-6)

    def test_analytical_gap_matches_paired_monte_carlo(self):
        c1=gap.s.C1;c0=-c1*gap.u_sep
        optimum=gap.argmin_objective(c0,c1)[0]
        expected=float(gap.objective(gap.u_sep,c0,c1)[2]-gap.objective(optimum,c0,c1)[2])
        costs=simulate_costs([gap.u_sep,optimum],c0,c1,samples=200000,seed=17)
        diff=costs[0]-costs[1]
        se=diff.std(ddof=1)/np.sqrt(len(diff))
        self.assertLess(abs(diff.mean()-expected),5*se)


class TeacherChecks(unittest.TestCase):
    def test_analytic_teacher_score_matches_log_density_gradient(self):
        mod=load_module('face_teacher_test',ROOT/'studies/dmd_pilot/teacher.py')
        rng=np.random.default_rng(12)
        z=rng.normal(size=(24,3)); labels=np.r_[np.zeros(12),np.ones(12)]
        teacher=mod.FaceTeacher(z,labels,n_atoms=8)
        x=rng.normal(size=(4,3)); t=.37
        centers=teacher.bank[teacher.half:]; cov=t*t+(1-t)**2*teacher.var
        def log_density(x):
            return logsumexp(-.5*np.sum((x[:,None,:]-(1-t)*centers)**2,axis=-1)/cov,axis=1)
        finite=np.empty_like(x);eps=1e-5
        for j in range(x.shape[1]):
            h=np.zeros_like(x);h[:,j]=eps
            finite[:,j]=(log_density(x+h)-log_density(x-h))/(2*eps)
        np.testing.assert_allclose(teacher.field(x,t,1.,score=True),finite,rtol=1e-6,atol=1e-7)

    @unittest.skipUnless(importlib.util.find_spec('sklearn'),'Install optional requirements for the NumPy MLP check')
    def test_manual_mlp_backprop_matches_finite_differences(self):
        mod=load_module('dmd_core_test',ROOT/'studies/dmd_pilot/dmd_core.py')
        rng=np.random.default_rng(8)
        network=mod.MLP(3,2,4,rng)
        x=rng.normal(size=(5,3)); upstream=rng.normal(size=(5,2))
        _,cache=network.forward(x)
        dx,grads=network.back(upstream,cache);eps=1e-6
        def loss():return float(np.sum(network.forward(x)[0]*upstream))
        for p,g in zip(network.p,grads):
            for idx in np.ndindex(p.shape):
                old=p[idx];p[idx]=old+eps;plus=loss();p[idx]=old-eps;minus=loss();p[idx]=old
                self.assertAlmostEqual(g[idx],(plus-minus)/(2*eps),places=6)
        for idx in np.ndindex(x.shape):
            old=x[idx];x[idx]=old+eps;plus=loss();x[idx]=old-eps;minus=loss();x[idx]=old
            self.assertAlmostEqual(dx[idx],(plus-minus)/(2*eps),places=6)


if __name__=='__main__':unittest.main()
