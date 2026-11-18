# Four-step distillation: an exploratory failure analysis

## Research question

Can intermediate teacher supervision and gradient truncation improve a four-step
student without losing variation in an attribute that is not explicitly requested?
The pilot conditions on smiles and tracks the eyeglasses attribute in CelebA.

This is a small latent-space experiment. `teacher.py` defines an analytic Gaussian
mixture teacher centered on training-image latents. Four residual NumPy MLP blocks
form the student; a learned auxiliary velocity field supplies the fake score.
PyTorch is used for the decoder and attribute reader. This is not a replication
of a large pretrained diffusion-model benchmark or a validated fairness method.

## What the supplied results support

The supplied w=1 confirmation tables contain three new training seeds, 53/71/89.
Mean effective nearest-center counts are about 384.23 for the multistep teacher,
267.69 for a direct four-step teacher, and 1.66 for the basic DMD student. The
anchor/truncation variants also collapse in this setting. Effective centers are an
entropy-based representation diagnostic, not a count of human identities.

The evaluator's balanced accuracy for eyeglasses among reconstructed smiling
validation images is 79.22%, below the prespecified 80% gate. Both guidance settings
therefore have `selected: null`. The results do not establish a winning cut point.
Reported attribute proportions are automatic predictions, not human annotations.
These tables were checked for internal consistency but full training was not rerun.

## Files and execution

`reported_results/` contains unchanged numeric tables from the supplied archive.
`RESULTS_original_zh.md` preserves the original interpretation; image references
there refer to the private original archive and are not bundled here.
No photos, trained weights or browser review gallery are distributed in this candidate.

After installing this directory's requirements, run scripts 01 through 04 from
this directory, then 06 for diversity diagnostics and 05 for human review:

```bash
python 01_prepare.py
python 02_learn.py
python 03_compare.py
python 04_confirm.py
python 06_diversity.py
python 05_review.py
```

Step 01 downloads an approximately 1.43 GB dataset archive. Read it before running;
obtain CelebA under its current research-use terms at
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html. The original script uses a Kaggle
mirror with a fixed archive checksum; availability was not tested in this audit.
Use Python 3.11+ (`hashlib.file_digest` is used). Training may take substantial time.

## Next experiment

First improve representation/decoder fidelity and validate the attribute reader;
then compare a stable baseline student against matched anchor-only and cut+anchor
variants. Keep data splits and seed lists fixed before evaluation. Report uncertainty
across independently trained students, training-time teacher calls, wall time, and
inference cost. Use fresh evaluation after changing the protocol.

Training batch count and inference step count do not equal total compute: additional
teacher anchors cost extra. More automatic smile predictions do not establish higher
quality when diversity has collapsed. Complete blind labels before using human-quality
claims, and preserve unreadable/uncertain responses rather than forcing labels.
