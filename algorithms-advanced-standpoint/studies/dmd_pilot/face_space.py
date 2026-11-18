"""Reserve PCA capacity for local facial structure rather than background/lighting."""
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter

class FaceSpace:
    def __init__(self,dim=96):
        self.model=PCA(n_components=dim,whiten=True,svd_solver='randomized',random_state=34)
        y,x=np.mgrid[:32,:32]
        # Fixed spatial weights on the aligned eye and mouth regions; no eyewear labels used.
        weight=1+4*np.exp(-((x-15.5)/10.)**4-((y-17)/4.)**4)+3*np.exp(-((x-15.5)/7.)**4-((y-24)/3.5)**4)
        self.weight=np.repeat(weight[:,:,None],3,axis=2).astype('float32')
    def features(self,flat):
        chunks=[]
        for start in range(0,len(flat),200):
            x=flat[start:start+200].reshape(-1,32,32,3)
            detail=x-gaussian_filter(x,sigma=(0,1.1,1.1,0))
            chunks.append(((x+2*detail)*self.weight).reshape(len(x),-1))
        return np.concatenate(chunks)
    def fit(self,x):
        self.model.fit(self.features(x))
        self.explained_variance_ratio_=self.model.explained_variance_ratio_
        # Linear least-squares reconstruction in unweighted pixel coordinates.
        z=self.transform(x)
        self.mean_=x.mean(axis=0)
        self.inverse_=np.linalg.lstsq(z,x-self.mean_,rcond=1.e-5)[0]
        return self
    def transform(self,x): return self.model.transform(self.features(x))
    def inverse_transform(self,z): return z@self.inverse_+self.mean_
