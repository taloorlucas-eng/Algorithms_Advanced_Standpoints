from study import *
from sklearn.metrics import balanced_accuracy_score,precision_score,recall_score

def main():
    exp=Study()
    choices=json.loads((OUT/'03_locked_choices.json').read_text())
    if any(c['fingerprint']!=exp.fingerprint for c in choices.values()):
        raise ValueError('Models/settings changed: rerun step 03 before step 04.')
    rows=[]; gallery={}; blind=[]
    x,y=load('test')
    ref=reference('test')
    labels=(probs(x,exp.judge)>=exp.reader['original_thresholds'])
    test_info={'reference':ref,'n_real_smiles':int(y[:,0].sum()),'n_real_smiles_glasses':int((y[:,0]*y[:,1]).sum()),
               'reader_B_given_A':float(labels[labels[:,0],1].mean()),
               'reader_bacc':[float(balanced_accuracy_score(y[:,i],labels[:,i])) for i in range(2)]}
    (OUT/'04_test_reference.json').write_text(json.dumps(test_info,indent=2))
    rng=np.random.default_rng(654)
    for w in GUIDANCE:
        chosen=choices[str(w)]['selected']
        # Always run predeclared diagnostic arms even if no policy qualifies.
        arms=list(dict.fromkeys(['teacher','DMD','cut1','anchor1','cut2','anchor2']+
                 ([chosen,'anchor'+chosen[3:]] if chosen and chosen.startswith('cut') else [])))
        for seed in CONFIRM:
            noise=np.random.default_rng(seed+300000).normal(size=(SIZE,exp.dim))
            for arm in arms:
                z=exp.generate(arm,w,seed,noise)
                scores,photos=exp.score(z,ref)
                rows.append(dict(w=w,seed=seed,arm=arm,reference=ref,**scores))
                if seed==CONFIRM[0]:
                    gallery[w,arm]=photos[:6]
                    for i in rng.choice(len(photos),15,replace=False):
                        blind.append((photos[i],w,arm,seed,int(i)))
            print(f'Test done: w={w:g} seed={seed}',flush=True)
        grid([gallery[w,arm] for arm in arms],arms,OUT/f'04_generated_w{w:g}.png',
             f'Generated only  |  smile guidance {w:g}  |  same noise in each column')
    write_rows('04_test.csv',rows)
    summary=[]
    for w,arm in gallery:
        rr=[r for r in rows if (r['w'],r['arm'])==(w,arm)]
        summary.append(dict(w=w,arm=arm,hit=means(rr,'hit'),glasses_given_smile=means(rr,'glasses_given_smile'),
                   mean_absolute_gap=means(rr,'conditional_gap'),sd_gap=float(np.std([r['conditional_gap'] for r in rr],ddof=1)),
                   mean_nn=means(rr,'mean_nn'),ref=ref))
    write_rows('04_summary.csv',summary)
    from PIL import Image
    audit=OUT/'blind'; audit.mkdir(exist_ok=True)
    key=[]; sheet=[]
    for j,idx in enumerate(rng.permutation(len(blind))):
        photo,w,arm,seed,i=blind[idx]; fn=f'{j+1:03}.png'
        Image.fromarray(photo).resize((192,192),Image.Resampling.BICUBIC).save(audit/fn)
        key.append(dict(file=fn,w=w,arm=arm,seed=seed,noise_index=i))
        sheet.append(dict(file=fn,smile='',eyeglasses='',readable=''))
    write_rows('04_blind_key.csv',key)
    # Protect entered human labels when the confirmation script is rerun.
    if not (OUT/'04_label_sheet.csv').exists(): write_rows('04_label_sheet.csv',sheet)
    for r in summary: print(r)
if __name__=='__main__': main()
