"""Shared experiment settings, paired-noise scoring, and durable per-arm training."""
from common import *
from types import SimpleNamespace
import io, csv, hashlib
import joblib
from sklearn.neighbors import NearestNeighbors
from dmd_core import train, rollout, MLP
from teacher import FaceTeacher

# Two separate guidance strengths; fix these before looking at results.
GUIDANCE=[1.,2.]
OPTIONS=['DMD','cut1','cut2','cut3','cut12','cut13','cut23','cut123']
DISCOVERY=[11,23,37]
CONFIRM=[53,71,89]
PARAMS=SimpleNamespace(hidden=96,steps=400,batch=48,fake_updates=2,substeps=2,lr=.001,lr_fake=.002,anchor=1.)
SIZE=1000

class Study:
    def __init__(self):
        setup()
        self.pca=joblib.load(WEIGHTS/'pca.joblib')
        self.decoder,self.judge=load_models()
        labels=load('train')[1]
        z=np.load(WEIGHTS/'training_latents.npy')
        self.teacher=FaceTeacher(z,labels[:,0])
        self.neighbors=NearestNeighbors(n_neighbors=1,algorithm='brute',n_jobs=1).fit(z)
        self.reader=json.loads((OUT/'02_reader.json').read_text())
        self.thresholds=self.reader['thresholds']
        self.dim=z.shape[1]
        digest=hashlib.sha256()
        for filename in [WEIGHTS/'training_latents.npy',WEIGHTS/'models.pt',BASE/'teacher.py',BASE/'dmd_core.py', BASE/'study.py']:
            with filename.open('rb') as fp:
                digest.update(hashlib.file_digest(fp,'sha256').digest())
        self.fingerprint=digest.hexdigest()[:16]
        self.cache=WEIGHTS/self.fingerprint
        self.cache.mkdir(exist_ok=True)
    def get_student(self,name,w,seed):
        path=self.cache/f'{name}-w{w:g}-s{seed}.npz'
        if path.exists():
            rng=np.random.default_rng(0)
            blocks=[MLP(self.dim,self.dim,PARAMS.hidden,rng) for _ in range(4)]
            with np.load(path) as f:
                for i,block in enumerate(blocks): block.p=[f[f'b{i}p{j}'] for j in range(4)]
            return blocks
        cuts=tuple(int(c) for c in name if c.isdigit())
        blocks=train(self.teacher,self.dim,cuts,w,seed,PARAMS,detach=not name.startswith('anchor'))
        tmp=path.with_suffix('.tmp.npz')
        np.savez(tmp,**{f'b{i}p{j}':p for i,b in enumerate(blocks) for j,p in enumerate(b.p)})
        os.replace(tmp,path)
        return blocks
    def generate(self,name,w,seed,noise):
        if name=='teacher':
            return np.concatenate([self.teacher.solve(noise[i:i+100],1.,0.,32,w) for i in range(0,len(noise),100)])
        return rollout(self.get_student(name,w,seed),noise)
    def score(self,z,ref):
        photos=decode(z,self.pca,self.decoder)
        pred=probs(photos,self.judge)>=self.thresholds
        smile=pred[:,0]; n=int(smile.sum()); k=int(pred[smile,1].sum())
        frac=k/n if n else float('nan')
        near=self.neighbors.kneighbors(z.astype('float32'))[0]
        return dict(generated=len(z),smile_count=n,glasses_and_smile=k,hit=n/len(z),glasses_given_smile=frac,
                    conditional_gap=abs(frac-ref) if n else 1.,mean_nn=float(near.mean())),photos
def reference(split):
    _,y=load(split)
    return float(y[y[:,0]==1,1].mean())
def write_rows(name,rows):
    with (OUT/name).open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
def means(rows,key):
    return float(np.mean([r[key] for r in rows]))
