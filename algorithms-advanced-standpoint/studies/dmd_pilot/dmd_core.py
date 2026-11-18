"""Four-block NumPy student, neural fake FM model, analytic linear-flow GMM teacher."""
import numpy as np
from scipy.special import softmax
from sklearn.mixture import GaussianMixture

class Teacher:
    def __init__(self, z, a, target, components):
        self.groups = []
        weights, means, variances = [], [], []
        for k in (0,1):
            subset = z[a==k]
            gm = GaussianMixture(components, covariance_type='diag', reg_covar=.025,
                                 random_state=17, max_iter=120).fit(subset)
            weights.extend(gm.weights_ * len(subset)/len(z))
            means.extend(gm.means_)
            variances.extend(gm.covariances_)
            self.groups.extend([k]*components)
        self.logw = np.log(weights)
        self.mu, self.var = np.asarray(means), np.asarray(variances)
        self.keep = np.array(self.groups)==target

    def field(self, x, t, guidance, score=False):
        # Linear Gaussian bridge Z_t=(1-t)X+t*noise; vectorized over mixture components.
        t = np.asarray(t).reshape(-1,1,1)
        a = 1-t
        cov = a*a*self.var + t*t
        delta = x[:,None,:] - a*self.mu
        logits = self.logw - .5*np.sum(delta*delta/cov + np.log(cov),axis=-1)
        cond = softmax(np.where(self.keep,logits,-np.inf),axis=1)
        post = cond if guidance==1 else softmax(logits,axis=1)*(1-guidance)+guidance*cond
        local = -delta/cov if score else -self.mu + (t-a*self.var)*delta/cov
        return np.einsum('nk,nkd->nd',post,local)

    def solve(self,x,start,end,n,guidance):
        # Midpoint integration (2 teacher evaluations per substep).
        h=(end-start)/n
        for i in range(n):
            t=start+i*h
            v=self.field(x,t,guidance)
            x=x+h*self.field(x+.5*h*v,t+.5*h,guidance)
        return x


class MLP:
    def __init__(self, dim, out, hidden, rng):
        self.p = [rng.normal(0,1/np.sqrt(dim),(dim,hidden)), np.zeros(hidden),
                  rng.normal(0,.015,(hidden,out)), np.zeros(out)]
        self.m = [np.zeros_like(p) for p in self.p]
        self.v = [np.zeros_like(p) for p in self.p]
        self.iter = 0

    def forward(self,x):
        w,b,v,c=self.p
        h=np.tanh(x@w+b)
        return h@v+c,(x,h)

    def back(self,grad,cache):
        x,h=cache
        dh=(grad@self.p[2].T)*(1-h*h)
        return dh@self.p[0].T,[x.T@dh,dh.sum(0),h.T@grad,grad.sum(0)]

    def update(self,grads,lr):
        self.iter+=1
        for i,g in enumerate(grads):
            self.m[i]=.9*self.m[i]+.1*g
            self.v[i]=.999*self.v[i]+.001*g*g
            self.p[i]-=lr*(self.m[i]/(1-.9**self.iter))/(np.sqrt(self.v[i]/(1-.999**self.iter))+1e-8)


def rollout(blocks, z, cache=False):
    states=[z]
    features=[]
    for block in blocks:
        dx, f=block.forward(z)
        z=z+dx
        states.append(z)
        features.append(f)
    return (states, features) if cache else z


def fake_features(z,t):
    return np.concatenate([z,t, np.sin(np.pi*t),np.cos(np.pi*t), z*t],axis=1)


def train(teacher, dim, cuts, w, seed, args, detach=True):
    rng=np.random.default_rng(seed)
    blocks=[MLP(dim,dim,args.hidden,rng) for _ in range(4)]
    fake=MLP(2*dim+3,dim,args.hidden,rng)
    for _ in range(args.steps):
        # Separate fresh minibatches for auxiliary and generator updates.
        for _aux in range(args.fake_updates):
            x=rollout(blocks,rng.normal(size=(args.batch,dim)))
            t=rng.uniform(.12,.88,(args.batch,1))
            eta=rng.normal(size=x.shape)
            z=(1-t)*x+t*eta
            base=(2*t-1)*z/((1-t)**2+t*t)  # exact velocity for a standard normal marginal
            pred,f=fake.forward(fake_features(z,t))
            gv=2*(base+pred-(eta-x))/(args.batch*dim)
            fake.update(fake.back(gv,f)[1],args.lr_fake)
        noise=rng.normal(size=(args.batch,dim))
        states,features=rollout(blocks,noise,cache=True)
        x=states[-1]
        t=rng.uniform(.12,.88,(args.batch,1))
        z=(1-t)*x+t*rng.normal(size=x.shape)
        vel=fake.forward(fake_features(z,t))[0]+(2*t-1)*z/((1-t)**2+t*t)
        fake_score=-(z+(1-t)*vel)/t
        teacher_score=teacher.field(z,t,w,score=True)
        # Gradient of a t-weighted noisy reverse KL. Factor (1-t) is dz/dx.
        downstream=(1-t)*t*(fake_score-teacher_score)/(args.batch*dim)
        anchors={}
        start=0
        for end in cuts:
            # Detached segment start. Separate first/middle segments receive supervision.
            target=teacher.solve(states[start],1-start/4,1-end/4,
                                 args.substeps*(end-start),w)
            anchors[end]=2*args.anchor*(states[end]-target)/(args.batch*dim*len(cuts))
            start=end
        for j in reversed(range(4)):
            # A cut discards the incoming downstream gradient; a local anchor replaces it.
            if j+1 in anchors:
                downstream=anchors[j+1] if detach else downstream+anchors[j+1]
            dx,grads=blocks[j].back(downstream,features[j])
            downstream=downstream+dx
            blocks[j].update(grads,args.lr)
    return blocks
