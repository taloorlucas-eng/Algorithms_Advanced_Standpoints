from pathlib import Path
import json
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='2'
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import importlib

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
OUT=BASE/'outputs'
WEIGHTS=BASE/'weights'
grid=importlib.import_module('01_prepare').grid
def device_name():
    # CPU is the reproducible default for the published pilot. Apple GPU is optional.
    return os.environ.get('FACE_DEVICE','cpu')

def setup():
    torch.set_num_threads(3)
    torch.manual_seed(34)
    OUT.mkdir(exist_ok=True); WEIGHTS.mkdir(exist_ok=True)

def load(split):
    with np.load(DATA/(split+'.npz')) as f: return f['x'],f['y']

def tensor(x):
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0,3,1,2).float().div(255)

def arr(t):
    return (t.detach().cpu().clamp(0,1).permute(0,2,3,1).numpy()*255).round().astype('uint8')

def pool(x):
    return x.reshape(-1,32,2,32,2,3).mean((2,4),dtype=np.float32).reshape(len(x),-1)/255

class Detail(nn.Module):
    """Learn a residual to restore details from the 32px PCA reconstruction."""
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,32,5,padding=2),nn.LeakyReLU(.1),
                 nn.Conv2d(32,32,3,padding=1),nn.LeakyReLU(.1),
                 nn.Conv2d(32,12,3,padding=1),nn.PixelShuffle(2))
    def forward(self,x):
        return F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)+self.conv(x)

class Judge(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        inc=3
        for ch in (24,32,48,64):
            layers.extend([nn.Conv2d(inc,ch,3,2,1),nn.BatchNorm2d(ch),nn.LeakyReLU(.1)])
            inc=ch
        self.net=nn.Sequential(*layers,nn.Flatten(),nn.Linear(64*4*4,96),nn.LeakyReLU(.1),nn.Dropout(.15),nn.Linear(96,2))
    def forward(self,x): return self.net(x)

def load_models():
    dev=device_name()
    dec,judge=Detail().to(dev),Judge().to(dev)
    state=torch.load(WEIGHTS/'models.pt',map_location=dev,weights_only=True)
    dec.load_state_dict(state['decoder']); judge.load_state_dict(state['judge'])
    return dec.eval(),judge.eval()

@torch.no_grad()
def decode(z,pca,dec):
    out=[]
    for chunk in np.array_split(z,max(1,int(np.ceil(len(z)/96)))):
        small=pca.inverse_transform(chunk.astype('float32')).reshape(-1,32,32,3)
        x=torch.from_numpy(small.copy()).permute(0,3,1,2).to(device_name())
        out.append(arr(dec(x)))
    return np.concatenate(out)

@torch.no_grad()
def probs(x,judge):
    return np.concatenate([judge(tensor(x[i:i+128]).to(device_name())).sigmoid().cpu().numpy()
                           for i in range(0,len(x),128)])
