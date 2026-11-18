"""Check whether many noise inputs end at the same face prototype."""
from study import *
from scipy.spatial.distance import cdist

def main():
    exp=Study()
    with (OUT/'04_test.csv').open() as fp: old=list(csv.DictReader(fp))
    for rr in list(old):
        if rr['arm']=='teacher': old.append(dict(rr,arm='coarse_teacher'))
    rows=[]; preview={}

    for rr in old:
        w=float(rr['w']);seed=int(rr['seed']);arm=rr['arm']
        noise=np.random.default_rng(seed+300000).normal(size=(SIZE,exp.dim))
        if arm=='coarse_teacher':
            # Four Euler calls to the original field. Keeps the teacher bank at inference.
            z=noise.copy()
            for j in range(4): z=z-.25*exp.teacher.field(z,1-j/4,w)
        else: z=exp.generate(arm,w,seed,noise)
        if seed==CONFIRM[0] and w==1 and arm in ('teacher','coarse_teacher','DMD','cut1','anchor1'):
            preview[arm]=decode(z[:6],exp.pca,exp.decoder)

        centers=exp.teacher.bank[exp.teacher.half:]
        closest=cdist(z,centers,metric='sqeuclidean').argmin(1)
        freq=np.bincount(closest,minlength=len(centers))/len(z)
        positive=freq[freq>0]
        rows.append(dict(w=w,seed=seed,arm=arm,n=len(z),distinct_nearest=int(len(positive)),
                    largest_cluster=float(positive.max()),effective_centers=float(np.exp(-sum(positive*np.log(positive)))),
                    latent_spread=float(np.sqrt(np.var(z,axis=0).mean()))))
    arms=['teacher','coarse_teacher','DMD','cut1','anchor1']
    grid([preview[arm] for arm in arms],['Teacher: 64 evals','Teacher: 4 evals','4-step DMD','Anchor 1 + cut','Anchor 1 only'],OUT/'06_same_noise.png',
         'Generated CelebA portraits | smile guidance w=1 | matched noise by column')
    write_rows('06_diversity.csv',rows)
    print('Saved outputs/06_diversity.csv. See largest_cluster, effective_centers and latent_spread.')
if __name__=='__main__': main()
