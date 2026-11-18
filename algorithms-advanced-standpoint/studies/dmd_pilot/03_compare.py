from study import *

def main():
    exp=Study(); rows=[]; policy={}
    ref=reference('val')
    diag=exp.reader
    reliable=all(diag[s]['smile']['balanced_acc']>=.8 and diag[s]['glasses_in_true_smiles']['balanced_acc']>=.8 and
                 diag[s]['glasses_in_true_smiles']['precision']>=.6 for s in ('validation_original','validation_reconstructed'))
    for w in GUIDANCE:
        for seed in DISCOVERY:
            noise=np.random.default_rng(seed+100000).normal(size=(SIZE,exp.dim))
            for arm in ['teacher']+OPTIONS:
                z=exp.generate(arm,w,seed,noise)
                scores,_=exp.score(z,ref)
                rows.append(dict(w=w,seed=seed,arm=arm,reference=ref,**scores))
            print(f'Validation done: w={w:g} seed={seed}',flush=True)
        def select(arm):
            return [r for r in rows if r['arm']==arm and r['w']==w]
        target_hit=max(.80,means(select('teacher'),'hit')-.03)
        max_nn=1.20*means(select('teacher'),'mean_nn')
        feasible=[arm for arm in OPTIONS if means(select(arm),'hit')>=target_hit and means(select(arm),'mean_nn')<=max_nn]
        best=min(feasible,key=lambda arm:means(select(arm),'conditional_gap')) if feasible and reliable else None
        policy[str(w)]={'fingerprint':exp.fingerprint,'selected':best,'reference':ref,'min_hit':target_hit,'max_nn':max_nn,'eligible':feasible,'reader_pass':reliable}
    write_rows('03_validation.csv',rows)
    (OUT/'03_locked_choices.json').write_text(json.dumps(policy,indent=2))
    print(json.dumps(policy,indent=2))
if __name__=='__main__': main()
