"""Download the public CelebA mirror, validate bytes, and cache a fixed subset."""
from pathlib import Path
import csv, hashlib, io, json, os, subprocess, zipfile
import numpy as np
from PIL import Image, ImageDraw
from collections import Counter

BASE=Path(__file__).resolve().parent
DATA=BASE/'data'
OUT=BASE/'outputs'
URL='https://www.kaggle.com/api/v1/datasets/download/jessicali9530/celeba-dataset'
HASH='4ceccf7dffb6f3b40559fb9ae68b42a71c3528b2881b787d65f4d1f75329f02d'
NAMES=['train','cal','val','test']

def download():
    DATA.mkdir(exist_ok=True)
    archive=DATA/'celeba.zip'
    if not archive.exists():
        tmp=DATA/'celeba.zip.part'
        print('Downloading CelebA (1.43 GB); interrupted downloads resume here.',flush=True)
        subprocess.run(['curl','-fL','--retry','3','-C','-','-o',str(tmp),URL],check=True)
        if not zipfile.is_zipfile(tmp): raise ValueError('Invalid download. See START_HERE.md.')
        os.replace(tmp,archive)
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=HASH:
            raise ValueError('Archive differs from the verified Kaggle version. See START_HERE.md.')
    return archive

def grid(rows, labels, path, title, cell=96):
    from PIL import ImageFont
    canvas=Image.new('RGB',(165+len(rows[0])*cell,52+len(rows)*cell),'white')
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.load_default(size=16)
    draw.text((8,14),title,font=font,fill='black')
    for i,(images,label) in enumerate(zip(rows,labels)):
        draw.text((8,52+i*cell+cell//2-10),label,font=font,fill='black')
        for j,im in enumerate(images):
            canvas.paste(Image.fromarray(im).resize((cell,cell),Image.Resampling.BICUBIC),(165+j*cell,52+i*cell))
    canvas.save(path)

def main():
    source=download()
    OUT.mkdir(exist_ok=True)
    rng=np.random.default_rng(20261106)
    with zipfile.ZipFile(source) as z:
        def read(name):
            return csv.DictReader(io.TextIOWrapper(z.open(name),encoding='utf8'))
        attrs={r['image_id']:[int(r['Smiling']=='1'),int(r['Eyeglasses']=='1')] for r in read('list_attr_celeba.csv')}
        groups=[[] for _ in range(3)]
        for row in read('list_eval_partition.csv'):
            groups[int(row['partition'])].append(row['image_id'])
        # Official validation photos split disjointly for calibration and mask selection.
        val=rng.choice(groups[1],4800,replace=False)
        lists=[rng.choice(groups[0],16000,replace=False),val[:2400],val[2400:],rng.choice(groups[2],2400,replace=False)]
        counts={}
        for name, ids in zip(NAMES,lists):
            xs=[]; ys=[]
            for fn in ids:
                with Image.open(io.BytesIO(z.read('img_align_celeba/img_align_celeba/'+fn))) as im:
                    xs.append(np.asarray(im.convert('RGB').crop((17,32,161,176)).resize((64,64),Image.Resampling.LANCZOS)))
                ys.append(attrs[fn])
            x,y=np.array(xs),np.array(ys,dtype=np.int64)
            tmp=DATA/(name+'.partial.npz')
            np.savez(tmp,x=x,y=y,ids=ids); os.replace(tmp,DATA/(name+'.npz'))
            counts[name]={f'smile{a}_glasses{b}':int(((y[:,0]==a)&(y[:,1]==b)).sum()) for a in (0,1) for b in (0,1)}
            print(name,len(y),counts[name],flush=True)
            if name=='val':
                grid([x[(y[:,0]==a)&(y[:,1]==b)][:5] for a,b in ((1,1),(1,0),(0,1),(0,0))],
                     ['Smile + glasses','Smile, no glasses','No smile + glasses','Neither'],OUT/'01_real_photos.png','Real CelebA photos grouped by dataset labels')
    (OUT/'01_counts.json').write_text(json.dumps(counts,indent=2))
if __name__=='__main__': main()
