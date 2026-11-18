"""Analytic linear-noising teacher. Each component is centred on a real training latent."""
import numpy as np
from scipy.special import softmax
class FaceTeacher:
    def __init__(self,z,smiling,n_atoms=512,sigma=.32):
        rng=np.random.default_rng(730)
        self.var=sigma**2
        self.bank=np.concatenate([z[rng.choice(np.flatnonzero(smiling==s),n_atoms,replace=False)] for s in (0,1)]).astype('float64')
        self.half=n_atoms
        self.prior=np.repeat(np.log([1-smiling.mean(),smiling.mean()]),n_atoms)
        self.norm=(self.bank*self.bank).sum(1)
    def field(self,x,t,guidance,score=False):
        t=np.asarray(t).reshape(-1,1)
        a=1-t
        cov=t*t+a*a*self.var
        # Common terms cancel in softmax.
        logits=(a*(x@self.bank.T)-.5*a*a*self.norm)/cov
        cond=softmax(logits[:,self.half:],axis=1)@self.bank[self.half:]
        if guidance!=1:
            uncond=softmax(logits+self.prior,axis=1)@self.bank
            cond=uncond+guidance*(cond-uncond)
        residual=x-a*cond
        if score: return -residual/cov
        return -cond+(t-a*self.var)*residual/cov
    def solve(self,x,start,end,n,guidance):
        h=(end-start)/n
        for i in range(n):
            t=start+i*h
            mid=x+.5*h*self.field(x,t,guidance)
            x=x+h*self.field(mid,t+.5*h,guidance)
        return x
