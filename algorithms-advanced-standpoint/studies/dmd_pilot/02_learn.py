"""Train a face representation, image decoder and separate attribute reader."""
from common import *
import joblib
from face_space import FaceSpace
from sklearn.metrics import balanced_accuracy_score,precision_score, recall_score, roc_auc_score

def main():
    setup()
    train,y=load('train')
    rng=np.random.default_rng(792)
    dev=device_name()
    small=pool(train)
    pca=FaceSpace().fit(small)
    z=pca.transform(small)
    bases=pca.inverse_transform(z).reshape(-1,32,32,3)
    joblib.dump(pca,WEIGHTS/'pca.joblib')
    dec,judge=Detail().to(dev),Judge().to(dev)
    optd=torch.optim.Adam(dec.parameters(),lr=.001)
    optj=torch.optim.AdamW(judge.parameters(),lr=.001)
    # Positive weighting is only for the rare-eyeglasses reader. Generation uses real prevalence.
    weights=torch.tensor([1.,3.],device=dev)
    for i in range(800):
        ids=rng.integers(len(y),size=64)
        clean=tensor(train[ids]).to(dev)
        xx=torch.from_numpy(bases[ids].copy()).permute(0,3,1,2).to(dev)
        yy=torch.from_numpy(y[ids]).float().to(dev)
        if i%2: xx=xx.flip([-1]);clean=clean.flip([-1])
        restored=dec(xx)
        loss=F.l1_loss(restored,clean)+.35*F.l1_loss(restored.diff(dim=-1),clean.diff(dim=-1))
        optd.zero_grad();loss.backward();optd.step()
        # The reader learns from both clean photos and reconstructed training photos.
        # No generated student samples enter reader training.
        reader_input=clean.clone()
        reader_input[::2]=restored.detach().clamp(0,1)[::2]
        out=judge(reader_input)
        lj=F.binary_cross_entropy_with_logits(out,yy,pos_weight=weights)
        optj.zero_grad();lj.backward();optj.step()
        if (i+1)%100==0: print(f'{i+1}/800  reconstruction {loss.item():.3f}  attributes {lj.item():.3f}',flush=True)
    torch.save({'decoder':dec.state_dict(),'judge':judge.state_dict()},WEIGHTS/'models.pt')
    dec.eval();judge.eval()
    cal,cy=load('cal'); cp=probs(cal,judge)
    # Threshold matches real attribute prevalence on independent calibration photos.
    thresholds=[float(np.quantile(cp[:,j],1-cy[:,j].mean())) for j in (0,1)]
    rp=probs(decode(pca.transform(pool(cal)),pca,dec),judge)
    recon_thresholds=[float(np.quantile(rp[:,j],1-cy[:,j].mean())) for j in (0,1)]
    def report(photos,labels,thresh):
        pp=probs(photos,judge); pred=pp>=thresh
        def stats(yy,p):
            return {'balanced_acc':float(balanced_accuracy_score(yy,p)), 'precision':float(precision_score(yy,p,zero_division=0)),
                    'recall':float(recall_score(yy,p,zero_division=0))}
        return {'smile':stats(labels[:,0],pred[:,0]),'glasses':stats(labels[:,1],pred[:,1]),
                'glasses_in_true_smiles':stats(labels[labels[:,0]==1,1],pred[labels[:,0]==1,1]),
                'true_B_given_A':float(labels[labels[:,0]==1,1].mean()),
                'pred_B_given_A':float(pred[pred[:,0],1].mean()) if pred[:,0].sum() else None}
    x,vy=load('val'); rz=decode(pca.transform(pool(x)),pca,dec)
    metrics={'thresholds':recon_thresholds,'original_thresholds':thresholds,'pca_variance':float(sum(pca.explained_variance_ratio_)),
             'validation_original':report(x,vy,thresholds),'validation_reconstructed':report(rz,vy,recon_thresholds)}
    (OUT/'02_reader.json').write_text(json.dumps(metrics,indent=2))
    # Show paired reconstruction examples, explicitly including the rare group.
    order=np.concatenate([np.flatnonzero((vy[:,0]==1)&(vy[:,1]==b))[:4] for b in (1,0)])
    grid([x[order],rz[order]],['Real photo','Reconstruction'],OUT/'02_reconstruction.png','Each column is the same held-out person and photo')
    # Save training latents; real test stays sealed until step 04.
    np.save(WEIGHTS/'training_latents.npy',z)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
